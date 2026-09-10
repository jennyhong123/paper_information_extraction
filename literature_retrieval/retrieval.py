"""
parsed_sections.json
      ↓
section_text를 max ~200 token으로 chunk
      ↓
각 chunk에 paper metadata 복사
      ↓
section_title + chunk_text → Embedding
      ↓
질문 Embedding → Cosine similarity → Top-k

python retrieval.py ../pmc_articles/PMC_UID_11774469/parsed_sections.json -k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config import FIELD_CONFIG

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
MIN_CHUNK_CHARS = 40

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_parsed_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _paper_meta(parsed: dict) -> Dict[str, str]:
    meta = parsed.get("metadata") or {}
    pmid = meta.get("pmid") or ""
    pmcid = meta.get("pmcid") or ""
    if pmid and pmcid:
        id_value = f"{pmid}/{pmcid}"
    else:
        id_value = pmid or pmcid
    authors = meta.get("authors") or []
    return {
        "pmc_uid": str(parsed.get("pmc_uid") or ""),
        "year": str(meta.get("year") or ""),
        "DOI": str(meta.get("doi") or ""),
        "PMID/PMCID": id_value,
        "title": str(meta.get("title") or ""),
        "journal": str(meta.get("journal") or ""),
        "author": authors[0] if authors else "",
    }


def collect_sections(parsed: dict) -> List[dict]:
    """abstract / introduction / discussion / all_sections / tables → 통일 section dict."""
    sections: List[dict] = []

    abstract = (parsed.get("abstract") or "").strip()
    if abstract:
        sections.append(
            {
                "section_title": "Abstract",
                "section_group": "abstract",
                "section_text": abstract,
            }
        )

    for key in ("introduction", "discussion", "all_sections"):
        for sec in parsed.get(key) or []:
            if not isinstance(sec, dict):
                continue
            text = (sec.get("section_text") or "").strip()
            if not text:
                continue
            sections.append(
                {
                    "section_title": sec.get("section_title") or "",
                    "section_group": sec.get("section_group") or "unknown",
                    "section_text": text,
                }
            )

    for table in parsed.get("tables") or []:
        if not isinstance(table, dict):
            continue
        body = _table_to_text(table)
        if not body:
            continue
        sections.append(
            {
                "section_title": table.get("label") or table.get("table_id") or "Table",
                "section_group": "table",
                "section_text": body,
            }
        )

    return sections


def _table_to_text(table: dict) -> str:
    """표 rows를 'col=value' 문장으로 변환."""
    caption = (table.get("caption") or "").strip()
    rows = table.get("rows") or []
    if not rows:
        text = (table.get("text") or "").strip()
        return " ".join(part for part in (caption, text) if part).strip()

    lines: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parts = [
            f"{k}={str(v).strip()}"
            for k, v in row.items()
            if str(v).strip()
        ]
        if parts:
            lines.append("; ".join(parts))
    body = " | ".join(lines)
    return " ".join(part for part in (caption, body) if part).strip()


def filter_sections(
    sections: Sequence[dict],
    allowed_groups: Optional[Iterable[str]] = None,
) -> List[dict]:
    if not allowed_groups:
        return list(sections)
    allowed = {g.lower() for g in allowed_groups}
    return [
        s
        for s in sections
        if (s.get("section_group") or "").lower() in allowed
    ]


def make_chunks(
    sections: Sequence[dict],
    paper_meta: Optional[Dict[str, str]] = None,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    model = get_model()
    tokenizer = model.tokenizer
    meta = paper_meta or {}
    chunks: List[dict] = []
    step = max(1, size - overlap)

    for sec in sections:
        text = (sec.get("section_text") or "").strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        title = sec.get("section_title") or ""
        group = sec.get("section_group") or "unknown"
        for i in range(0, len(ids), step):
            chunk_text = tokenizer.decode(ids[i : i + size]).strip()
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                {
                    **meta,
                    "section_group": group,
                    "section_title": title,
                    "text": chunk_text,
                }
            )
            if i + size >= len(ids):
                break
    return chunks


def chunks_from_parsed(
    parsed: dict,
    section_groups: Optional[Iterable[str]] = None,
    size: int = CHUNK_SIZE,
) -> List[dict]:
    sections = filter_sections(collect_sections(parsed), section_groups)
    return make_chunks(sections, paper_meta=_paper_meta(parsed), size=size)


def embed_chunks(chunks: Sequence[dict]) -> np.ndarray:
    model = get_model()
    texts = [
        f"{c.get('section_title', '')} {c.get('text', '')}".strip()
        for c in chunks
    ]
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return model.encode(texts, normalize_embeddings=True)


def retrieve(
    query: str,
    chunks: Sequence[dict],
    chunk_emb: Optional[np.ndarray] = None,
    top_k: int = 3,
) -> List[dict]:
    model = get_model()
    if chunk_emb is None:
        chunk_emb = embed_chunks(chunks)
    if len(chunks) == 0:
        return []
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = chunk_emb @ query_emb
    k = min(top_k, len(chunks))
    top_idx = np.argsort(scores)[::-1][:k]
    results: List[dict] = []
    for i in top_idx:
        item = dict(chunks[int(i)])
        item["score"] = float(scores[int(i)])
        results.append(item)
    return results


def retrieve_field(parsed: dict, field: str, top_k: int = 3) -> List[dict]:
    if field not in FIELD_CONFIG:
        raise KeyError(
            f"Unknown field: {field}. Choose from {list(FIELD_CONFIG)}"
        )
    cfg = FIELD_CONFIG[field]
    chunks = chunks_from_parsed(parsed, section_groups=cfg.get("sections"))
    return retrieve(cfg["query"], chunks, top_k=top_k)


def retrieve_all_fields(parsed: dict, top_k: int = 3) -> Dict[str, List[dict]]:
    return {field: retrieve_field(parsed, field, top_k=top_k) for field in FIELD_CONFIG}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk/embed parsed_sections.json and retrieve"
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to parsed_sections.json",
    )
    parser.add_argument(
        "-f",
        "--field",
        default=None,
        choices=list(FIELD_CONFIG),
        help="Single FIELD_CONFIG key (default: run all fields)",
    )
    parser.add_argument("-k", "--top-k", type=int, default=3)
    args = parser.parse_args()

    parsed = load_parsed_json(args.json_path)
    fields = [args.field] if args.field else list(FIELD_CONFIG)
    for field in fields:
        hits = retrieve_field(parsed, field, top_k=args.top_k)
        cfg = FIELD_CONFIG[field]
        print("=" * 70)
        print(f"field={field}")
        print(f"query={cfg['query']}")
        print(f"sections={cfg['sections']}")
        print(f"hits={len(hits)}")
        for i, hit in enumerate(hits, start=1):
            print(
                f"\n[{i} score={hit.get('score', 0):.4f}] "
                f"{hit.get('section_group')} | {hit.get('section_title')}"
            )
            print(hit.get("text", ""))


if __name__ == "__main__":
    main()
