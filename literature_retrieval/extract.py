"""
RAG extraction over parsed_sections.json using retrieval + Ollama.

Examples:
  python extract.py ../pmc_articles/PMC_UID_11774469/parsed_sections.json
  python extract.py ../pmc_articles --csv ../pmc_articles/article_information.csv
  python extract.py ../pmc_articles --no-score
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import FIELD_CONFIG
from ollama_client import OllamaClient, OllamaConfig
from retrieval import load_parsed_json, retrieve_field
from score import score_extract_row

SYSTEM_PROMPT = (
    "You extract study facts from biomedical paper excerpts.\n"
    "Answer using ONLY the provided context snippets.\n"
    "If the answer is not clearly supported, reply exactly: NOT_FOUND\n"
    "Do not invent numbers, cohorts, assays, or biomarkers.\n"
    "Keep the answer concise (1-4 sentences)."
)

FIELD_COLUMNS = list(FIELD_CONFIG.keys())
CSV_COLUMNS = ["Paper_ID", "PMCID"] + FIELD_COLUMNS


def lookup_paper_id(
    pmcid: str,
    metadata_csv: Path,
) -> str:
    """Map PMCID -> Paper_ID using article_metadata.csv when available."""
    if not pmcid or not metadata_csv.is_file():
        return ""
    try:
        df = pd.read_csv(metadata_csv)
    except Exception:
        return ""
    if "Paper_ID" not in df.columns or "PMID/PMCID" not in df.columns:
        return ""
    target = pmcid.upper().replace("PMC", "").strip()
    for _, row in df.iterrows():
        combined = str(row.get("PMID/PMCID") or "")
        # Prefer PMC-looking token; else last slash segment.
        parts = [p.strip() for p in combined.split("/") if p.strip()]
        pmc_token = next((p for p in parts if p.upper().startswith("PMC")), None)
        if pmc_token is None and parts:
            pmc_token = parts[-1]
        if not pmc_token:
            continue
        if pmc_token.upper().replace("PMC", "").strip() == target:
            return str(row.get("Paper_ID") or "")
    return ""


def _pmcid_from_parsed(parsed: dict) -> str:
    meta = parsed.get("metadata") or {}
    pmcid = meta.get("pmcid") or ""
    if pmcid:
        return pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    uid = parsed.get("pmc_uid") or ""
    return f"PMC{uid}" if uid else ""


def _format_context(hits: List[dict], max_chars: int = 3500) -> str:
    parts: List[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        block = (
            f"[{i}] ({hit.get('section_group')}) {hit.get('section_title')}\n"
            f"{hit.get('text', '')}"
        )
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def extract_field(
    parsed: dict,
    field: str,
    client: OllamaClient,
    top_k: int = 4,
) -> str:
    cfg = FIELD_CONFIG[field]
    hits = retrieve_field(parsed, field, top_k=top_k)
    context = _format_context(hits)
    if not context.strip():
        return "NOT_FOUND"
    prompt = (
        f"Field: {field}\n"
        f"Question: {cfg['query']}\n\n"
        f"Context snippets:\n{context}\n\n"
        "Answer:"
    )
    text = (client.generate(prompt, system=SYSTEM_PROMPT) or "").strip()
    return text or "NOT_FOUND"


def extract_article(
    parsed: dict,
    client: OllamaClient,
    metadata_csv: Optional[Path] = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    pmcid = _pmcid_from_parsed(parsed)
    paper_id = ""
    if metadata_csv is not None:
        paper_id = lookup_paper_id(pmcid, metadata_csv)
    row: Dict[str, Any] = {
        "Paper_ID": paper_id,
        "PMCID": pmcid,
    }
    for field in FIELD_COLUMNS:
        print(f"  extracting {field}...")
        row[field] = extract_field(parsed, field, client, top_k=top_k)
    return row


def _iter_parsed_paths(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    paths = sorted(path.glob("PMC_UID_*/parsed_sections.json"))
    if not paths and (path / "parsed_sections.json").is_file():
        paths = [path / "parsed_sections.json"]
    return paths


def upsert_information_csv(row: Dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for existing in reader:
                if existing.get("PMCID") == row.get("PMCID"):
                    continue
                rows.append(existing)
    rows.append({k: row.get(k, "") for k in CSV_COLUMNS})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract study fields from parsed PMC JSON via RAG + Ollama"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="parsed_sections.json or directory containing PMC_UID_*/parsed_sections.json",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output for a single article",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="article_information.csv path (default: ../pmc_articles/article_information.csv)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="article_metadata.csv for Paper_ID lookup",
    )
    parser.add_argument("-m", "--model", default="qwen3.5:9b")
    parser.add_argument("-k", "--top-k", type=int, default=4)
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip scoring after extraction",
    )
    parser.add_argument(
        "--score-csv",
        type=Path,
        default=None,
        help="article_score.csv path (default: ../pmc_articles/article_score.csv)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "pmc_articles"
    info_csv = args.csv or (root / "article_information.csv")
    metadata_csv = args.metadata or (root / "article_metadata.csv")
    score_csv = args.score_csv or (root / "article_score.csv")

    client = OllamaClient(OllamaConfig(model=args.model, thinking=False))
    if not client.is_available():
        raise SystemExit(
            f"Ollama not available. Start with: ollama serve && ollama pull {args.model}"
        )

    paths = _iter_parsed_paths(args.path)
    if not paths:
        raise SystemExit(f"No parsed_sections.json found under {args.path}")

    for json_path in paths:
        print(f"extracting {json_path}")
        parsed = load_parsed_json(json_path)
        row = extract_article(
            parsed,
            client,
            metadata_csv=metadata_csv,
            top_k=args.top_k,
        )
        if not row.get("Paper_ID"):
            # Fallback: keep empty Paper_ID; caller can backfill later
            pass
        upsert_information_csv(row, info_csv)
        print(f"  saved -> {info_csv} ({row.get('Paper_ID')} {row.get('PMCID')})")

        if args.out and len(paths) == 1:
            args.out.write_text(
                json.dumps(row, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  json -> {args.out}")

        if not args.no_score:
            score_extract_row(row, score_csv, model=args.model)
            print(f"  scored -> {score_csv}")


if __name__ == "__main__":
    main()
