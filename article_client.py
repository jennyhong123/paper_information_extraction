"""PMC article client — wraps article_collector for the MCI literature pipeline."""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Optional

import requests

from base import BaseSourceClient, ExternalArticle
from article_collector import (
    extract_oa_package,
    find_tgz_url,
    parse_pmc_xml_file,
    parse_pmc_xml_text,
    write_parsed_json,
)

logger = logging.getLogger(__name__)

""" Function :
1.PMC article 수집
2.PMC article 자료 다운로드
3.PMC XML 파싱
"""


class PmcArticleClient(BaseSourceClient):
    """PubMed Central client using NCBI E-utilities."""

    EMAIL = "jinhee.hong@cellkey.co.kr"
    NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # =========================
    # 1. NCBI E-utilities request
    # =========================
    @staticmethod
    def _ncbi_get(endpoint: str, params: dict, max_retries: int = 3) -> requests.Response:
        """NCBI E-utilities 요청. 500/502/503/504 서버 오류는 재시도한다."""
        params = {
            **params,
            "tool": "pmc_collect_parse_download",
            "email": PmcArticleClient.EMAIL,
        }
        url = f"{PmcArticleClient.NCBI_BASE}/{endpoint}"

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                time.sleep(0.34)
                return response
            except requests.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None

                print(f"NCBI request failed: attempt {attempt}/{max_retries}")
                print(f"Status code: {status_code}")
                print(f"URL: {e.response.url if e.response is not None else url}")

                if status_code in {500, 502, 503, 504} and attempt < max_retries:
                    time.sleep(2 * attempt)
                    continue
                raise

    def _search_pmc(
        self,
        query: str,
        retmax: int = 5,
        year_from: Optional[int] = None,
    ) -> List[str]:
        """Search PMC and return UID list. Prints total hit count."""
        params = {
            "db": "pmc",
            "term": query,
            "retmode": "json",
            "retmax": str(retmax),
        }
        # NCBI publication-date filter: year_from and later (pdat).
        if year_from is not None:
            params["mindate"] = str(year_from)
            params["maxdate"] = "3000"
            params["datetype"] = "pdat"

        response = self._ncbi_get("esearch.fcgi", params)
        result = response.json().get("esearchresult", {})
        total = result.get("count", "?")
        ids = result.get("idlist", [])
        print(f"PMC esearch total count: {total} (returning {len(ids)})")
        logger.info("PMC esearch total count=%s returned=%d", total, len(ids))
        return ids

    def search(
        self,
        query: str,
        max_results: int = 20,
        year_from: Optional[int] = None,
    ) -> List[str]:
        pmc_uids = self._search_pmc(query, retmax=max_results, year_from=year_from)
        logger.info("Found %d PMC articles", len(pmc_uids))
        return pmc_uids

    def _fetch_pmc_xml(self, pmc_uid: str) -> str:
        """PMC UID로 full-text XML 가져오기."""
        response = self._ncbi_get(
            "efetch.fcgi",
            {
                "db": "pmc",
                "id": str(pmc_uid),
                "retmode": "xml",
            },
        )
        return response.text

    # =========================
    # 2. File download helper
    # =========================

    def _download_file(
        self, url: str, out_path: Path, force: bool = False, timeout: int = 120
    ) -> Path:
        """URL 파일 다운로드. 기존 파일이 있으면 기본적으로 skip."""
        out_path = Path(out_path)
        if out_path.is_file() and not force:
            logger.info("Skip existing file: %s", out_path)
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

        size = out_path.stat().st_size
        logger.info("Downloaded file: %s (%d bytes)", out_path, size)
        return out_path

    def download_xml(self, pmc_uid: str, article_dir: Path, force: bool = False) -> Path:
        """PMC XML을 article_dir/article.xml로 다운로드하고 기존 파일은 skip."""
        article_dir.mkdir(parents=True, exist_ok=True)
        xml_path = article_dir / "article.xml"
        if xml_path.is_file() and not force:
            logger.info("Skip existing XML: %s", xml_path)
            return xml_path

        xml_text = self._fetch_pmc_xml(pmc_uid)
        if not xml_text:
            raise ValueError(f"Empty PMC XML for UID {pmc_uid}")

        xml_path.write_text(xml_text, encoding="utf-8")
        logger.info("Saved XML: %s", xml_path)
        return xml_path

    def download_oa_package(self, pmcid: str, article_dir: Path, force: bool = False) -> dict:
        """PMCID의 OA package를 다운로드/압축 해제하고 상태 dict를 반환."""
        oa_package_info = {
            "oa_tgz_url": "",
            "oa_tgz_downloaded": False,
            "extracted_files": [],
        }
        if not pmcid:
            return oa_package_info

        try:
            tgz_url = find_tgz_url(pmcid)
            oa_package_info["oa_tgz_url"] = tgz_url
            print(tgz_url, "\n--------------------------------\n")
            if not tgz_url:
                logger.info("No OA package URL for %s", pmcid)
                return oa_package_info

            package_dir = article_dir / "oa_package"
            tgz_path = package_dir / "oa_package.tar.gz"
            extracted_files = (
                [
                    str(path)
                    for path in package_dir.glob("*")
                    if path.is_file() and path.name != "oa_package.tar.gz"
                ]
                if package_dir.is_dir()
                else []
            )

            if tgz_path.is_file() and extracted_files and not force:
                logger.info("Skip existing OA package: %s", package_dir)
                oa_package_info["oa_tgz_downloaded"] = True
                oa_package_info["oa_tgz_path"] = str(tgz_path)
                oa_package_info["extracted_files"] = extracted_files
                oa_package_info["skipped_existing"] = True
                return oa_package_info

            self._download_file(tgz_url, tgz_path, force=force, timeout=120)
            extracted = extract_oa_package(tgz_path, package_dir)
            oa_package_info["oa_tgz_downloaded"] = True
            oa_package_info["oa_tgz_path"] = str(tgz_path)
            oa_package_info["extracted_files"] = extracted
        except Exception as e:
            oa_package_info["oa_error"] = str(e)
            logger.warning("OA package download failed for %s: %s", pmcid, e)

        return oa_package_info

    def collect_pmc_uid(
        self,
        pmc_uid: str,
        out_dir: Path,
        force: bool = False,
    ) -> dict:
        """단일 PMC UID의 XML/OA 다운로드를 제어하고 parsed JSON을 저장.

        OA package가 없어도 article_dir을 삭제하지 않고 XML 파싱을 계속한다.
        """
        article_dir = Path(out_dir) / f"PMC_UID_{pmc_uid}"
        article_dir.mkdir(parents=True, exist_ok=True)

        xml_path = self.download_xml(pmc_uid, article_dir, force=force)
        parsed = parse_pmc_xml_file(xml_path, pmc_uid=pmc_uid)
        metadata = parsed["metadata"]
        pmcid = metadata.get("pmcid", "")

        oa_package_info = self.download_oa_package(pmcid, article_dir, force=force)
        oa_skipped = not bool(oa_package_info.get("oa_tgz_downloaded"))
        if oa_skipped:
            oa_package_info["oa_skipped"] = True
            logger.info(
                "OA package unavailable for PMC UID %s (pmcid=%s); continuing with XML",
                pmc_uid,
                pmcid or "(none)",
            )

        parsed["oa_package_info"] = oa_package_info
        parsed_path = write_parsed_json(parsed, article_dir)
        article = self._fetch_one(pmc_uid, parsed=parsed)

        logger.info(
            "Saved parsed JSON for PMC UID %s: %s (sections=%d tables=%d figures=%d oa_skipped=%s)",
            pmc_uid,
            parsed_path,
            len(parsed["all_sections"]),
            len(parsed["tables"]),
            len(parsed["figures"]),
            oa_skipped,
        )

        return {
            "pmc_uid": pmc_uid,
            "pmcid": pmcid,
            "article_dir": str(article_dir),
            "xml_path": str(xml_path),
            "parsed_json_path": str(parsed_path),
            "oa_package_info": oa_package_info,
            "oa_skipped": oa_skipped,
            "metadata": metadata,
            "article": article,
        }

    def collect_to_disk(
        self,
        query: str,
        out_dir: Path,
        max_results: int = 20,
        force: bool = False,
        year_from: Optional[int] = None,
        on_collected: Optional[Callable[[dict, int], None]] = None,
    ) -> List[dict]:
        """검색 후 각 PMC UID를 out_dir에 XML/OA/parsed JSON으로 저장한다."""
        out_dir.mkdir(exist_ok=True)
        collected = []
        for pmc_uid in self.search(query, max_results=max_results, year_from=year_from):
            logger.info("Processing PMC UID: %s", pmc_uid)
            try:
                result = self.collect_pmc_uid(pmc_uid, out_dir, force=force)
                collected.append(result)
                if on_collected:
                    on_collected(result, len(collected))
            except Exception as e:
                logger.warning("Failed to collect PMC UID %s: %s", pmc_uid, e)
        return collected

    # =========================
    # 3. PMC article parsing
    # =========================

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
                xml_text = self._fetch_pmc_xml(pmc_uid)
                parsed = parse_pmc_xml_text(xml_text, pmc_uid=pmc_uid)
        except ET.ParseError as e:
            logger.warning("Invalid XML for PMC UID %s: %s", pmc_uid, e)
            return None
        except requests.HTTPError as e:
            logger.warning("Failed to fetch PMC XML for %s: %s", pmc_uid, e)
            return None

        meta = parsed["metadata"]
        pmid = meta.get("pmid", "")
        pmcid = meta.get("pmcid", "")
        doi = meta.get("doi") or None

        if pmid:
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif pmcid:
            link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        else:
            link = ""

        article = ExternalArticle(
            source_id=pmid or pmcid or str(pmc_uid),
            source="pmc",
            title=meta.get("title", ""),
            abstract=parsed.get("abstract", "") or "",
            authors=meta.get("authors", []),
            year=meta.get("year", ""),
            link=link,
            journal=meta.get("journal", ""),
            doi=doi,
            pmcid=pmcid or None,
        )
        print(article, "\n--------------------------------\n")
        return article


def _write_metadata_csv(collected: List[dict], out_path: Path) -> None:
    """Save article_metadata.csv with Paper_ID P001... columns."""
    fieldnames = [
        "Paper_ID",
        "DB",
        "year",
        "DOI",
        "PMID/PMCID",
        "title",
        "journal",
        "author",
        "source_url",
    ]
    rows = []
    for idx, item in enumerate(collected, start=1):
        article = item.get("article")
        if article is None:
            continue
        meta = article.to_metadata(query="")
        rows.append({"Paper_ID": f"P{idx:03d}", **meta})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metadata CSV: {out_path} ({len(rows)} rows)")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Collect PMC articles to disk")
    parser.add_argument("query", help="PubMed/PMC search query")
    parser.add_argument("-o", "--out-dir", default="./pmc_articles", help="Output directory")
    parser.add_argument("-n", "--max-results", type=int, default=20, help="Max PMC results")
    parser.add_argument("-y", "--year", type=int, default=None, help="Publication year from (mindate)")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    client = PmcArticleClient()
    collected = client.collect_to_disk(
        query=args.query,
        out_dir=out_dir,
        max_results=args.max_results,
        force=args.force,
        year_from=args.year,
    )
    _write_metadata_csv(collected, out_dir / "article_metadata.csv")
    print(f"Collected {len(collected)} articles -> {out_dir}")


if __name__ == "__main__":
    main()
