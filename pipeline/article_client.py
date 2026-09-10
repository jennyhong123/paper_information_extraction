"""PMC article client — search, paginate, download, parse."""

from __future__ import annotations

import argparse
import logging
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

from .article_collector import (
    extract_oa_package,
    find_tgz_url,
    parse_pmc_xml_file,
    parse_pmc_xml_text,
    write_parsed_json,
)
from .common import (
    BaseSourceClient,
    ExternalArticle,
    append_jsonl,
    assign_paper_ids,
    atomic_write_json,
    load_paper_id_map,
    normalize_pmcid,
    sha256_file,
    utc_now,
    write_csv_atomic,
)

logger = logging.getLogger(__name__)


class PmcArticleClient(BaseSourceClient):
    NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        email: Optional[str] = None,
        tool: str = "pmc_collect_parse_download",
        api_key: Optional[str] = None,
    ):
        self.email = email or os.environ.get("NCBI_EMAIL", "jinhee.hong@cellkey.co.kr")
        self.tool = tool
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")

    def _ncbi_get(self, endpoint: str, params: dict, max_retries: int = 3) -> requests.Response:
        params = {**params, "tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.NCBI_BASE}/{endpoint}"
        sleep_s = 0.1 if self.api_key else 0.34

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                time.sleep(sleep_s)
                return response
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status in {500, 502, 503, 504} and attempt < max_retries:
                    time.sleep(2 * attempt)
                    continue
                raise

    def search_all_uids(
        self,
        query: str,
        year_from: Optional[int] = None,
        page_size: int = 200,
        max_retries: int = 3,
    ) -> Dict:
        """Paginate esearch and return sorted unique PMC UIDs."""
        page_size = max(1, min(int(page_size), 500))
        params = {
            "db": "pmc",
            "term": query,
            "retmode": "json",
            "retmax": str(page_size),
            "retstart": "0",
        }
        if year_from is not None:
            params["mindate"] = str(year_from)
            params["maxdate"] = "3000"
            params["datetype"] = "pdat"

        first = self._ncbi_get("esearch.fcgi", params, max_retries=max_retries).json()
        result = first.get("esearchresult", {})
        total = int(result.get("count") or 0)
        ids: List[str] = list(result.get("idlist") or [])
        retstart = page_size
        while retstart < total:
            params["retstart"] = str(retstart)
            page = self._ncbi_get("esearch.fcgi", params, max_retries=max_retries).json()
            batch = page.get("esearchresult", {}).get("idlist") or []
            if not batch:
                break
            ids.extend(batch)
            retstart += page_size

        unique = sorted(set(str(i) for i in ids), key=lambda x: int(x) if x.isdigit() else x)
        return {"esearch_total": total, "pmc_uids": unique, "returned": len(unique)}

    def search(
        self,
        query: str,
        max_results: int = 20,
        year_from: Optional[int] = None,
    ) -> List[str]:
        found = self.search_all_uids(query, year_from=year_from, page_size=min(max_results, 500))
        return found["pmc_uids"][:max_results]

    def _fetch_pmc_xml(self, pmc_uid: str) -> str:
        response = self._ncbi_get(
            "efetch.fcgi",
            {"db": "pmc", "id": str(pmc_uid), "retmode": "xml"},
        )
        return response.text

    def _download_file(
        self, url: str, out_path: Path, force: bool = False, timeout: int = 120
    ) -> Path:
        out_path = Path(out_path)
        if out_path.is_file() and not force:
            return out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["wget", "-q", "-O", str(out_path), url],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            out_path.unlink(missing_ok=True)
            err = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"wget failed (exit {result.returncode}): {err}")
        return out_path

    def download_xml(self, pmc_uid: str, article_dir: Path, force: bool = False) -> Path:
        article_dir.mkdir(parents=True, exist_ok=True)
        xml_path = article_dir / "article.xml"
        if xml_path.is_file() and not force:
            return xml_path
        xml_text = self._fetch_pmc_xml(pmc_uid)
        if not xml_text:
            raise ValueError(f"Empty PMC XML for UID {pmc_uid}")
        xml_path.write_text(xml_text, encoding="utf-8")
        return xml_path

    def download_oa_package(self, pmcid: str, article_dir: Path, force: bool = False) -> dict:
        info = {"oa_tgz_url": "", "oa_tgz_downloaded": False, "extracted_files": []}
        if not pmcid:
            return info
        try:
            tgz_url = find_tgz_url(pmcid)
            info["oa_tgz_url"] = tgz_url
            if not tgz_url:
                return info
            package_dir = article_dir / "oa_package"
            tgz_path = package_dir / "oa_package.tar.gz"
            extracted = (
                [
                    str(p)
                    for p in package_dir.glob("*")
                    if p.is_file() and p.name != "oa_package.tar.gz"
                ]
                if package_dir.is_dir()
                else []
            )
            if tgz_path.is_file() and extracted and not force:
                info.update(
                    {
                        "oa_tgz_downloaded": True,
                        "oa_tgz_path": str(tgz_path),
                        "extracted_files": extracted,
                        "skipped_existing": True,
                    }
                )
                return info
            self._download_file(tgz_url, tgz_path, force=force)
            extracted = extract_oa_package(tgz_path, package_dir)
            info.update(
                {
                    "oa_tgz_downloaded": True,
                    "oa_tgz_path": str(tgz_path),
                    "extracted_files": extracted,
                }
            )
        except Exception as e:
            info["oa_error"] = str(e)
            logger.warning("OA package failed for %s: %s", pmcid, e)
        return info

    def repair_parsed(self, pmc_uid: str, out_dir: Path) -> Optional[dict]:
        article_dir = Path(out_dir) / f"PMC_UID_{pmc_uid}"
        xml_path = article_dir / "article.xml"
        parsed_path = article_dir / "parsed_sections.json"
        if not xml_path.is_file() or parsed_path.is_file():
            return None
        parsed = parse_pmc_xml_file(xml_path, pmc_uid=pmc_uid)
        write_parsed_json(parsed, article_dir)
        return parsed

    def collect_pmc_uid(self, pmc_uid: str, out_dir: Path, force: bool = False) -> dict:
        article_dir = Path(out_dir) / f"PMC_UID_{pmc_uid}"
        article_dir.mkdir(parents=True, exist_ok=True)
        xml_path = self.download_xml(pmc_uid, article_dir, force=force)
        parsed_path = article_dir / "parsed_sections.json"
        if parsed_path.is_file() and not force:
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        else:
            parsed = parse_pmc_xml_file(xml_path, pmc_uid=pmc_uid)

        metadata = parsed["metadata"]
        pmcid = metadata.get("pmcid", "")
        oa = self.download_oa_package(pmcid, article_dir, force=force)
        parsed["oa_package_info"] = oa
        write_parsed_json(parsed, article_dir)
        article = self._fetch_one(pmc_uid, parsed=parsed)
        return {
            "pmc_uid": pmc_uid,
            "pmcid": normalize_pmcid(pmcid) or normalize_pmcid(f"PMC{pmc_uid}"),
            "article_dir": str(article_dir),
            "xml_path": str(xml_path),
            "xml_sha256": sha256_file(xml_path),
            "parsed_json_path": str(article_dir / "parsed_sections.json"),
            "parsed_sha256": sha256_file(article_dir / "parsed_sections.json"),
            "oa_package_info": oa,
            "oa_skipped": not bool(oa.get("oa_tgz_downloaded")),
            "metadata": metadata,
            "article": article,
            "status": "ok",
        }

    def collect_uids(
        self,
        pmc_uids: List[str],
        out_dir: Path,
        force: bool = False,
        status_log: Optional[Path] = None,
        dry_run: bool = False,
    ) -> List[dict]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        collected: List[dict] = []
        for pmc_uid in pmc_uids:
            article_dir = out_dir / f"PMC_UID_{pmc_uid}"
            xml_path = article_dir / "article.xml"
            parsed_path = article_dir / "parsed_sections.json"
            if dry_run:
                needs = force or not (xml_path.is_file() and parsed_path.is_file())
                collected.append(
                    {
                        "pmc_uid": pmc_uid,
                        "status": "needs_work" if needs else "ok",
                        "dry_run": True,
                    }
                )
                continue
            try:
                if xml_path.is_file() and not parsed_path.is_file():
                    self.repair_parsed(pmc_uid, out_dir)
                result = self.collect_pmc_uid(pmc_uid, out_dir, force=force)
                collected.append(result)
                if status_log:
                    append_jsonl(
                        status_log,
                        {
                            "ts": utc_now(),
                            "stage": "collect",
                            "pmc_uid": pmc_uid,
                            "pmcid": result.get("pmcid"),
                            "status": "ok",
                        },
                    )
            except Exception as e:
                logger.warning("Failed %s: %s", pmc_uid, e)
                row = {"pmc_uid": pmc_uid, "status": "failed", "error": str(e)}
                collected.append(row)
                if status_log:
                    append_jsonl(
                        status_log,
                        {"ts": utc_now(), "stage": "collect", **row},
                    )
        return collected

    def collect_to_disk(
        self,
        query: str,
        out_dir: Path,
        max_results: int = 20,
        force: bool = False,
        year_from: Optional[int] = None,
        on_collected: Optional[Callable[[dict, int], None]] = None,
    ) -> List[dict]:
        uids = self.search(query, max_results=max_results, year_from=year_from)
        collected = []
        for item in self.collect_uids(uids, out_dir, force=force):
            if item.get("status") == "ok":
                collected.append(item)
                if on_collected:
                    on_collected(item, len(collected))
        write_metadata_csv(collected, Path(out_dir) / "article_metadata.csv")
        return collected

    def fetch_details(self, ids: List[str]) -> List[ExternalArticle]:
        articles: List[ExternalArticle] = []
        for pmc_uid in ids:
            article = self._fetch_one(pmc_uid)
            if article:
                articles.append(article)
        return articles

    def _fetch_one(self, pmc_uid: str, parsed: Optional[dict] = None) -> Optional[ExternalArticle]:
        try:
            if parsed is None:
                parsed = parse_pmc_xml_text(self._fetch_pmc_xml(pmc_uid), pmc_uid=pmc_uid)
        except (ET.ParseError, requests.HTTPError) as e:
            logger.warning("Failed fetch %s: %s", pmc_uid, e)
            return None
        meta = parsed["metadata"]
        pmid = meta.get("pmid", "")
        pmcid = meta.get("pmcid", "")
        if pmid:
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif pmcid:
            link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        else:
            link = ""
        return ExternalArticle(
            source_id=pmid or pmcid or str(pmc_uid),
            source="pmc",
            title=meta.get("title", ""),
            abstract=parsed.get("abstract", "") or "",
            authors=meta.get("authors", []),
            year=meta.get("year", ""),
            link=link,
            journal=meta.get("journal", ""),
            doi=meta.get("doi") or None,
            pmcid=pmcid or None,
        )


def write_metadata_csv(collected: List[dict], out_path: Path) -> Dict[str, str]:
    existing = load_paper_id_map(out_path)
    pmcids = []
    by_pmcid: Dict[str, dict] = {}
    for item in collected:
        if item.get("status") not in (None, "ok"):
            continue
        article = item.get("article")
        pmcid = normalize_pmcid(item.get("pmcid") or "")
        if not pmcid or article is None:
            continue
        pmcids.append(pmcid)
        by_pmcid[pmcid] = item
    id_map = assign_paper_ids(pmcids, existing)
    rows = []
    for pmcid in sorted(by_pmcid):
        article = by_pmcid[pmcid]["article"]
        meta = article.to_metadata(query="")
        rows.append({"Paper_ID": id_map[pmcid], **meta})
    write_csv_atomic(
        out_path,
        [
            "Paper_ID",
            "DB",
            "year",
            "DOI",
            "PMID/PMCID",
            "title",
            "journal",
            "author",
            "source_url",
        ],
        rows,
    )
    return id_map


def freeze_uid_manifest(
    out_dir: Path,
    run_id: str,
    query: str,
    year_from: Optional[int],
    search_result: Dict,
) -> Path:
    manifests = Path(out_dir) / "manifests"
    path = manifests / f"{run_id}.json"
    atomic_write_json(
        path,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "query": query,
            "year_from": year_from,
            "esearch_total": search_result["esearch_total"],
            "pmc_uids": search_result["pmc_uids"],
            "returned": search_result["returned"],
        },
    )
    return path


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Collect PMC articles to disk")
    parser.add_argument("query", help="PubMed/PMC search query")
    parser.add_argument("-o", "--out-dir", default="./pmc_articles")
    parser.add_argument("-n", "--max-results", type=int, default=20)
    parser.add_argument("-y", "--year", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Collect all query hits")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    client = PmcArticleClient()
    out_dir = Path(args.out_dir)
    if args.all:
        found = client.search_all_uids(args.query, year_from=args.year)
        uids = found["pmc_uids"]
        print(f"esearch total={found['esearch_total']} unique={len(uids)}")
    else:
        uids = client.search(args.query, max_results=args.max_results, year_from=args.year)
    collected = client.collect_uids(uids, out_dir, force=args.force)
    ok = [c for c in collected if c.get("status") == "ok"]
    write_metadata_csv(ok, out_dir / "article_metadata.csv")
    print(f"Collected {len(ok)}/{len(collected)} articles -> {out_dir}")


if __name__ == "__main__":
    main()
