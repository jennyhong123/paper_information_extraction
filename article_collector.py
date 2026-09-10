import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import requests

"""
과정: 
step1. keyword query로 논문 후보 검색 
step2. PMID, DOI, PMCID 저장
step3. PMCID가 있으면 Europe PMC or PMC에서 XML 다운로드
step4. XML parser로 abstract/body/table/fig/supplementary 추출
    TODO table 포맷을 LLM이 잘 인식하도록 수정해야됨
    
step5. figure caption (이건 나중에 생각하기)
step6. Supplementary-material에서 xlsx/csv/pdf 링크 저장
"""

# =========================
# step 1. Query들 (query문)
# =========================


PMC_OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi" # PMC OA API URL

RETMAX = 100
SUPP_TABLE_EXTENSIONS = (".xlsx", ".xls", ".csv", ".tsv")

'''
XML가 다운로드 불가능한 경우들이 있음 
 -> 이 경우 다른 방법으로 다운로드 받거나
 -> 일단 건너뛰기 (다른 경우로 다운로드 받거나)
'''

def check_pmc_xml_status(xml_text):
    # PMC 결과 XML안에 내용 여부 확인 
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {
            "pmcid": None,
            "pmid": None,
            "doi": None,
            "has_abstract": False,
            "has_body": False,
            "has_full_text_xml": False,
        }

    article = find_article(root)
    metadata = parse_metadata(article)
    has_abstract = any(local_name(elem.tag) == "abstract" for elem in article.iter())
    has_body = any(local_name(elem.tag) == "body" for elem in article.iter())

    return {
        "pmcid": metadata.get("pmcid") or None,
        "pmid": metadata.get("pmid") or None,
        "doi": metadata.get("doi") or None,
        "has_abstract": has_abstract,
        "has_body": has_body,
        "has_full_text_xml": has_body,
    }

# =========================
# 2. XML helper
# =========================

def local_name(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split())


def elem_text(elem):
    if elem is None:
        return ""
    return clean_text(" ".join(elem.itertext()))


def get_first_text(root, tag_name):
    for elem in root.iter():
        if local_name(elem.tag) == tag_name:
            text = elem_text(elem)
            if text:
                return text
    return ""


def _journal_meta(article):
    return next((e for e in article.iter() if local_name(e.tag) == "journal-meta"), None)


def _journal_title(journal_meta) -> str:
    if journal_meta is None:
        return ""
    for elem in journal_meta.iter():
        if local_name(elem.tag) == "journal-title":
            text = elem_text(elem)
            if text:
                return text
    return ""


def _journal_abbrev(journal_meta) -> str:
    if journal_meta is None:
        return ""
    for id_type in ("nlm-ta", "iso-abbrev", "publisher-id"):
        for elem in journal_meta.iter():
            if local_name(elem.tag) != "journal-id":
                continue
            if elem.attrib.get("journal-id-type", "") != id_type:
                continue
            text = clean_text(elem.text)
            if text:
                return text
    return ""


def get_href(elem):
    """xlink:href / href 모두 처리
    # XML element의 attribute 중 href를 찾는 것으로 
    주로 figure image, PDF, supplementary file 링크가 여기에 들어감
    """
    for key, value in elem.attrib.items():
        if local_name(key) == "href":
            return value
    return ""


def find_article(root):

    if local_name(root.tag) == "article":
        return root

    for elem in root.iter():
        if local_name(elem.tag) == "article":
            return elem

    return root


# =========================
# 3. XML parsing
# =========================

def parse_metadata(article):
    jmeta = _journal_meta(article)
    journal_title = _journal_title(jmeta)
    journal_name = _journal_abbrev(jmeta) or journal_title
    metadata = {
        "title": get_first_text(article, "article-title"),
        "journal": journal_title or journal_name,
        "journal_title": journal_title,
        "journal_name": journal_name,
        "publication_date": "",
        "year": get_first_text(article, "year"),
        "pmid": "",
        "pmcid": "",
        "doi": "",
    }

    pub_dates: dict[str, str] = {}
    for elem in article.iter():
        tag = local_name(elem.tag)
        if tag == "pub-date":
            pub_type = elem.attrib.get("pub-type", "") or "unknown"
            pub_dates[pub_type] = _format_pub_date(elem)
        elif tag == "article-id":
            id_type = elem.attrib.get("pub-id-type", "")
            value = clean_text(elem.text)

            if id_type == "pmid":
                metadata["pmid"] = value
            elif id_type in ("pmc", "pmcid"):
                metadata["pmcid"] = value if value.startswith("PMC") else f"PMC{value}"
            elif id_type == "doi":
                metadata["doi"] = value

    for pub_type in ("epub", "ppub", "collection", "unknown"):
        if pub_dates.get(pub_type):
            metadata["publication_date"] = pub_dates[pub_type]
            break
    if not metadata["publication_date"] and pub_dates:
        metadata["publication_date"] = next(iter(pub_dates.values()), "")

    metadata["authors"] = parse_authors(article)
    return metadata


def _format_pub_date(elem) -> str:
    parts: dict[str, str] = {}
    for child in elem:
        name = local_name(child.tag)
        if name in ("year", "month", "day"):
            parts[name] = clean_text(child.text)
    year = parts.get("year", "")
    if not year:
        return elem_text(elem)
    month = (parts.get("month") or "01").zfill(2)
    day = (parts.get("day") or "01").zfill(2)
    if parts.get("month") or parts.get("day"):
        return f"{year}-{month}-{day}"
    return year


def parse_authors(article):
    """article-meta 안의 contrib-type=author만 추출."""
    article_meta = next((e for e in article.iter() if local_name(e.tag) == "article-meta"), None)
    if not article_meta:
        return []

    authors = []
    for contrib in article_meta.iter():
        if local_name(contrib.tag) != "contrib" or contrib.attrib.get("contrib-type") != "author":
            continue
        collab = next((c for c in contrib if local_name(c.tag) == "collab"), None)
        if collab is not None:
            authors.append(elem_text(collab))
            continue
        name = next((c for c in contrib if local_name(c.tag) == "name"), None)
        if name is not None:
            full = clean_text(f"{get_first_text(name, 'given-names')} {get_first_text(name, 'surname')}".strip())
            if full:
                authors.append(full)
    return authors


def parse_abstract(article):
    for elem in article.iter():
        if local_name(elem.tag) == "abstract":
            return elem_text(elem)
    return ""


def infer_section_group(title, sec_type="", parent_group=None):
    text = f"{title} {sec_type}".lower()

    if "intro" in text or "background" in text:
        return "introduction"
    if "method" in text or "material" in text or "experimental" in text:
        return "methods"
    if "result" in text:
        return "results"
    if "discussion" in text:
        return "discussion"
    if parent_group:
        return parent_group
    return "unknown"


def direct_section_text(sec):
    """현재 sec 바로 아래 text만. nested sec·table-wrap 제외."""
    parts = []
    for child in list(sec):
        name = local_name(child.tag)
        if name in {"title", "sec", "table-wrap"}:
            continue
        parts.append(elem_text(child))
    return clean_text(" ".join(parts))


def get_section_title(sec):
    for child in list(sec):
        if local_name(child.tag) == "title":
            return elem_text(child)
    return ""


def parse_sec_recursive(sec, parent_title="", parent_group=None):
    title = get_section_title(sec)
    sec_type = sec.attrib.get("sec-type", "")
    section_group = infer_section_group(title=title, sec_type=sec_type, parent_group=parent_group)
    text = direct_section_text(sec)

    sections = []
    if title or text:
        sections.append({
            "section_title": title,
            "section_group": section_group,
            "parent_title": parent_title,
            "section_text": text,
        })

    for child in list(sec):
        if local_name(child.tag) == "sec":
            sections.extend(
                parse_sec_recursive(
                    child,
                    parent_title=title or parent_title,
                    parent_group=section_group,
                )
            )
    return sections


def parse_sections(article):
    sections = []
    for body in article.iter():
        if local_name(body.tag) != "body":
            continue
        for child in list(body):
            if local_name(child.tag) == "sec":
                sections.extend(parse_sec_recursive(child))
    return sections


def pick_sections(sections, keywords):
    picked = []

    for section in sections:
        title = section["section_title"].lower()

        if any(keyword.lower() in title for keyword in keywords):
            picked.append(section)

    return picked


def first_child(elem, tag_name):
    for child in list(elem):
        if local_name(child.tag) == tag_name:
            return child
    return None


def get_cells(tr):
    return [
        elem_text(cell)
        for cell in list(tr)
        if local_name(cell.tag) in {"td", "th"}
    ]


def merge_continuation_rows(columns, rows):
    """첫 cell이 비어 있으면 이전 row continuation으로 병합."""
    records = []
    current = None

    for row in rows:
        row = row + [""] * (len(columns) - len(row))
        if row[0].strip():
            current = {columns[i]: row[i] for i in range(len(columns))}
            records.append(current)
        elif current:
            for i, value in enumerate(row):
                if value.strip():
                    col = columns[i]
                    current[col] = clean_text(f"{current.get(col, '')} {value}")

    return records


def parse_tables(article):
    parsed_tables = []

    for table_wrap in article.iter():
        if local_name(table_wrap.tag) != "table-wrap":
            continue

        label = elem_text(first_child(table_wrap, "label"))
        caption = elem_text(first_child(table_wrap, "caption"))
        table = first_child(table_wrap, "table")
        if table is None:
            continue

        columns = []
        thead = first_child(table, "thead")
        if thead is not None:
            for tr in thead.iter():
                if local_name(tr.tag) == "tr":
                    cells = get_cells(tr)
                    if cells:
                        columns = cells

        body_rows = []
        tbody = first_child(table, "tbody")
        row_parent = tbody if tbody is not None else table

        for tr in row_parent.iter():
            if local_name(tr.tag) == "tr":
                cells = get_cells(tr)
                if cells:
                    body_rows.append(cells)

        if not columns and body_rows:
            columns = body_rows[0]
            body_rows = body_rows[1:]

        rows = merge_continuation_rows(columns, body_rows)

        parsed_tables.append({
            "table_id": label.rstrip(".") or table_wrap.attrib.get("id", ""),
            "xml_id": table_wrap.attrib.get("id", ""),
            "label": label,
            "caption": caption,
            "columns": columns,
            "rows": rows,
            "text": clean_text(" ".join(" ".join(r) for r in body_rows)),
        })

    return parsed_tables


def parse_figures(article):
    # figure 내용 parsing하기
    figures = []

    for fig in article.iter():
        if local_name(fig.tag) != "fig":
            continue

        item = {
            "figure_id": fig.attrib.get("id", ""),
            "label": "",
            "caption": "",
            "graphic_refs": [],
        }

        for child in list(fig):
            name = local_name(child.tag)

            if name == "label":
                item["label"] = elem_text(child)
            elif name == "caption":
                item["caption"] = elem_text(child)
            elif name == "graphic":
                href = get_href(child)
                if href:
                    item["graphic_refs"].append(href)

        figures.append(item)

    return figures


_FIGURE_LABEL_NUM_RE = re.compile(r"(?:figure|fig\.?)\s*(\d+)", re.IGNORECASE)
_FIGURE_ID_NUM_RE = re.compile(r"^F(\d+)$", re.IGNORECASE)


def _article_figure_key(figure: dict, idx: int) -> str:
    label = (figure.get("label") or "").strip()
    m = _FIGURE_LABEL_NUM_RE.search(label)
    if m:
        return f"figure{m.group(1)}"
    figure_id = (figure.get("figure_id") or "").strip()
    m = _FIGURE_ID_NUM_RE.match(figure_id)
    if m:
        return f"figure{m.group(1)}"
    return f"figure{idx}"


def _article_figure_dict(figures: list) -> dict[str, list[str]]:
    figure: dict[str, list[str]] = {}
    for idx, item in enumerate(figures, start=1):
        if not isinstance(item, dict):
            continue
        text = " ".join(part for part in (item.get("label"), item.get("caption")) if part).strip()
        if not text:
            continue
        figure.setdefault(_article_figure_key(item, idx), []).append(text[:1200])
    return figure


def _article_figure_files(figures: list) -> dict[str, str]:
    files: dict[str, str] = {}
    for idx, item in enumerate(figures, start=1):
        if not isinstance(item, dict):
            continue
        refs = [str(ref).strip() for ref in (item.get("graphic_refs") or []) if str(ref).strip()]
        if not refs:
            continue
        files[_article_figure_key(item, idx)] = refs[0] if len(refs) == 1 else ", ".join(refs)
    return files


def _selected_figures(
    figure: dict[str, list[str]],
    figure_files: dict[str, str] | None = None,
) -> dict[str, str]:
    """Selected figure keys mapped to graphic file names."""
    if not isinstance(figure, dict) or not figure:
        return {}

    def _sort_key(key: str) -> tuple:
        m = re.match(r"figure(\d+)", key, re.IGNORECASE)
        return (0, int(m.group(1))) if m else (1, key)

    selected: dict[str, str] = {}
    for key in sorted(figure.keys(), key=_sort_key):
        selected[key] = (figure_files or {}).get(key, "")
    return selected


def parse_supplementary_materials(article):
    supplementary_files = []

    for elem in article.iter():
        if local_name(elem.tag) != "supplementary-material":
            continue

        # 1. supplementary-material 자체에서 href 찾기
        href = get_href(elem)

        # 2. 없으면 내부 태그(media, graphic 등)에서 href 찾기
        if not href:
            for child in elem.iter():
                if child is elem:
                    continue

                child_href = get_href(child)
                if child_href:
                    href = child_href
                    break

        item = {
            "supplementary_id": elem.attrib.get("id", ""),
            "content_type": elem.attrib.get("content-type", ""),
            "label": "",
            "caption": "",
            "href": href,
        }

        # 3. label, caption도 내부 전체에서 찾기
        for child in elem.iter():
            name = local_name(child.tag)

            if name == "label":
                item["label"] = elem_text(child)
            elif name == "caption":
                item["caption"] = elem_text(child)

        supplementary_files.append(item)

    return supplementary_files
def parse_pmc_xml_text(xml_text: str, pmc_uid: str = "", oa_package_info=None) -> dict:
    """PMC XML 문자열을 parsed_sections.json 구조의 dict로 변환."""
    root = ET.fromstring(xml_text)
    article = find_article(root)

    metadata = parse_metadata(article)
    sections = parse_sections(article)
    tables = parse_tables(article)
    figures = parse_figures(article)
    supplementary_files = parse_supplementary_materials(article)
    figure = _article_figure_dict(figures)
    figure_files = _article_figure_files(figures)

    introduction = pick_sections(sections, ["intro", "background"])
    discussion = pick_sections(sections, ["discussion", "conclusion"])
    picked = {id(s) for s in introduction + discussion}

    return {
        "pmc_uid": pmc_uid,
        "metadata": metadata,
        "abstract": parse_abstract(article),
        "introduction": introduction,
        "discussion": discussion,
        "all_sections": [s for s in sections if id(s) not in picked],
        "tables": tables,
        "figures": figures,
        "figure": figure,
        "selected_figure": _selected_figures(figure, figure_files),
        "supplementary_files": supplementary_files,
        "oa_package_info": oa_package_info or {
            "oa_tgz_url": "",
            "oa_tgz_downloaded": False,
            "extracted_files": [],
        },
    }


def parse_pmc_xml_file(xml_path, pmc_uid: str = "", oa_package_info=None) -> dict:
    """PMC XML 파일을 읽어서 parsed_sections.json 구조의 dict로 변환."""
    xml_text = Path(xml_path).read_text(encoding="utf-8")
    return parse_pmc_xml_text(xml_text, pmc_uid=pmc_uid, oa_package_info=oa_package_info)


def write_parsed_json(parsed: dict, article_dir) -> Path:
    """parsed_sections.json 저장."""
    parsed_path = Path(article_dir) / "parsed_sections.json"
    parsed_path.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return parsed_path


# =========================
# 4. OA API / extract helper
# =========================


def normalize_download_url(url: str) -> str:
    """ftp.ncbi 링크를 https로 변환"""
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return url.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")
    return url


def _deprecated_oa_package_url(url: str) -> str:
    """OA API가 주는 oa_package 경로가 404일 때 deprecated 경로로 변환"""
    marker = "/pub/pmc/oa_package/"
    deprecated = "/pub/pmc/deprecated/oa_package/"
    if marker in url and deprecated not in url:
        return url.replace(marker, deprecated)
    return url


def _url_exists(url: str, timeout: int = 30) -> bool:
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 405:
            response = requests.get(url, timeout=timeout, stream=True)
            response.close()
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_pmc_oa_links(pmcid: str, timeout: int = 60) -> dict:
    """
    PMC OA API에서 PDF / TGZ 링크 조회.

    Returns:
        {"pdf": str, "tgz": str, "all": list[dict]}
    """
    response = requests.get(PMC_OA_API, params={"id": pmcid}, timeout=timeout)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    links = {"pdf": "", "tgz": "", "all": []}

    for elem in root.iter():
        if elem.tag != "link" and not elem.tag.endswith("}link"):
            continue

        fmt = elem.attrib.get("format", "").lower()
        href = elem.attrib.get("href", "")
        if not href:
            continue

        href = normalize_download_url(href)
        links["all"].append({"format": fmt, "href": href})

        if fmt == "pdf":
            links["pdf"] = href
        elif fmt in {"tgz", "tar.gz"}:
            links["tgz"] = href

    return links


def find_tgz_url(pmcid: str, timeout: int = 60) -> str:
    """PMCID에 대한 OA package(.tar.gz) 다운로드 URL 반환. 없으면 빈 문자열."""
    try:
        links = get_pmc_oa_links(pmcid, timeout=timeout)
    except requests.RequestException:
        return ""

    tgz_url = links["tgz"]
    if not tgz_url:
        return ""

    candidates = [tgz_url]
    deprecated_url = _deprecated_oa_package_url(tgz_url)
    if deprecated_url != tgz_url:
        candidates.append(deprecated_url)

    for url in candidates:
        if _url_exists(url, timeout=timeout):
            return url

    return ""



def extract_oa_package(tgz_path, package_dir):
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["tar", "-xzf", str(tgz_path), "-C", str(package_dir),"--strip-components=1",],
        check=True,
    )

    extracted = [
        str(path)
        for path in package_dir.rglob("*")
        if path.is_file() and path != Path(tgz_path)
    ]

    return extracted

