"""RAG extraction: retrieval + Ollama field extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .common import (
    FIELD_CONFIG,
    QUERIES_PATH,
    OllamaClient,
    OllamaConfig,
    append_jsonl,
    atomic_write_json,
    load_paper_id_map,
    normalize_pmcid,
    sha256_file,
    sha256_text,
    utc_now,
    write_csv_atomic,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
MIN_CHUNK_CHARS = 40

_model: Any = None
_model_name: Optional[str] = None


def configure_retrieval(
    embedding_model: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> None:
    global MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, _model, _model_name
    if embedding_model and embedding_model != MODEL_NAME:
        MODEL_NAME = embedding_model
        _model = None
        _model_name = None
    if chunk_size is not None:
        CHUNK_SIZE = int(chunk_size)
    if chunk_overlap is not None:
        CHUNK_OVERLAP = int(chunk_overlap)


def get_model():
    global _model, _model_name
    if _model is None or _model_name != MODEL_NAME:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
        _model_name = MODEL_NAME
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

SYSTEM_PROMPT = (
    "You extract study facts from biomedical paper excerpts.\n"
    "Answer using ONLY the provided context snippets.\n"
    "If the answer is not clearly supported, reply exactly: NOT_FOUND\n"
    "Do not invent numbers, cohorts, assays, or biomarkers.\n"
    "Keep the answer concise (1-4 sentences)."
)

FIELD_COLUMNS = list(FIELD_CONFIG.keys())
CSV_COLUMNS = ["Paper_ID", "PMCID"] + FIELD_COLUMNS


def _pmcid_from_parsed(parsed: dict) -> str:
    meta = parsed.get("metadata") or {}
    pmcid = meta.get("pmcid") or ""
    if pmcid:
        return normalize_pmcid(pmcid)
    uid = parsed.get("pmc_uid") or ""
    return normalize_pmcid(f"PMC{uid}") if uid else ""


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


def extract_fingerprint(
    parsed_path: Path,
    model: str,
    seed: int,
    temperature: float,
    top_k: int,
) -> str:
    payload = {
        "parsed_sha256": sha256_file(parsed_path),
        "queries_sha256": sha256_file(QUERIES_PATH),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "model": model,
        "seed": seed,
        "temperature": temperature,
        "top_k": top_k,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def extract_field(
    parsed: dict,
    field: str,
    client: OllamaClient,
    top_k: int = 4,
) -> Dict[str, Any]:
    cfg = FIELD_CONFIG[field]
    hits = retrieve_field(parsed, field, top_k=top_k)
    context = _format_context(hits)
    if not context.strip():
        return {"answer": "NOT_FOUND", "hits": []}
    prompt = (
        f"Field: {field}\n"
        f"Question: {cfg['query']}\n\n"
        f"Context snippets:\n{context}\n\n"
        "Answer:"
    )
    text = (client.generate(prompt, system=SYSTEM_PROMPT) or "").strip() or "NOT_FOUND"
    return {
        "answer": text,
        "hits": [
            {
                "section_group": h.get("section_group"),
                "section_title": h.get("section_title"),
                "score": h.get("score"),
                "text": h.get("text"),
            }
            for h in hits
        ],
    }


def extract_article(
    parsed: dict,
    client: OllamaClient,
    paper_id: str = "",
    top_k: int = 4,
) -> Dict[str, Any]:
    pmcid = _pmcid_from_parsed(parsed)
    fields: Dict[str, Any] = {}
    answers: Dict[str, str] = {"Paper_ID": paper_id, "PMCID": pmcid}
    for field in FIELD_COLUMNS:
        result = extract_field(parsed, field, client, top_k=top_k)
        fields[field] = result
        answers[field] = result["answer"]
    return {"answers": answers, "fields": fields}


def make_extract_client(
    model: str = "qwen3.5:9b",
    seed: int = 42,
    temperature: float = 0.0,
) -> OllamaClient:
    return OllamaClient(
        OllamaConfig(
            model=model,
            thinking=False,
            temperature=temperature,
            options={"seed": seed},
        )
    )


def _iter_parsed_paths(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    paths = sorted(path.glob("PMC_UID_*/parsed_sections.json"))
    if not paths and (path / "parsed_sections.json").is_file():
        return [path / "parsed_sections.json"]
    return paths


def run_extract_batch(
    root: Path,
    metadata_csv: Path,
    model: str = "qwen3.5:9b",
    seed: int = 42,
    temperature: float = 0.0,
    top_k: int = 4,
    force: bool = False,
    dry_run: bool = False,
    max_retries: int = 2,
    status_log: Optional[Path] = None,
    client: Optional[OllamaClient] = None,
) -> Dict[str, Any]:
    root = Path(root)
    id_map = load_paper_id_map(metadata_csv)
    paths = _iter_parsed_paths(root)
    stats = {"ok": 0, "skipped": 0, "failed": 0, "needs_work": 0}
    if not dry_run:
        client = client or make_extract_client(model=model, seed=seed, temperature=temperature)
        if not client.is_available():
            raise RuntimeError(f"Ollama not available for model={model}")
        model_digest = client.model_digest(model)
    else:
        model_digest = ""

    for parsed_path in paths:
        out_path = parsed_path.parent / "extraction.json"
        fp = extract_fingerprint(parsed_path, model, seed, temperature, top_k)
        if out_path.is_file() and not force:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("status") == "ok" and prev.get("fingerprint") == fp:
                stats["skipped"] += 1
                continue
        if dry_run:
            stats["needs_work"] += 1
            continue

        parsed = load_parsed_json(parsed_path)
        pmcid = _pmcid_from_parsed(parsed)
        paper_id = id_map.get(pmcid, "")
        last_err = ""
        for _ in range(max_retries):
            try:
                result = extract_article(parsed, client, paper_id=paper_id, top_k=top_k)
                atomic_write_json(
                    out_path,
                    {
                        "status": "ok",
                        "fingerprint": fp,
                        "pmcid": pmcid,
                        "paper_id": paper_id,
                        "parsed_sha256": sha256_file(parsed_path),
                        "model": model,
                        "model_digest": model_digest,
                        "seed": seed,
                        "temperature": temperature,
                        "top_k": top_k,
                        "ts": utc_now(),
                        **result,
                    },
                )
                stats["ok"] += 1
                if status_log:
                    append_jsonl(
                        status_log,
                        {"ts": utc_now(), "stage": "extract", "pmcid": pmcid, "status": "ok"},
                    )
                last_err = ""
                break
            except Exception as e:
                last_err = str(e)
        if last_err:
            atomic_write_json(
                out_path,
                {
                    "status": "failed",
                    "fingerprint": fp,
                    "pmcid": pmcid,
                    "paper_id": paper_id,
                    "error": last_err,
                    "ts": utc_now(),
                },
            )
            stats["failed"] += 1
            if status_log:
                append_jsonl(
                    status_log,
                    {
                        "ts": utc_now(),
                        "stage": "extract",
                        "pmcid": pmcid,
                        "status": "failed",
                        "error": last_err,
                    },
                )

    if not dry_run:
        aggregate_information_csv(root, root / "article_information.csv")
    return stats


def aggregate_information_csv(root: Path, csv_path: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(root).glob("PMC_UID_*/extraction.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "ok":
            continue
        answers = data.get("answers") or {}
        rows.append({k: answers.get(k, "") for k in CSV_COLUMNS})
    rows.sort(key=lambda r: str(r.get("PMCID") or ""))
    write_csv_atomic(csv_path, CSV_COLUMNS, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract study fields via RAG + Ollama")
    parser.add_argument("path", type=Path, help="pmc_articles dir or parsed_sections.json")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("-m", "--model", default="qwen3.5:9b")
    parser.add_argument("-k", "--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.path if args.path.is_dir() else args.path.parent.parent
    if args.path.is_file() and args.path.parent.name.startswith("PMC_UID_"):
        root = args.path.parent.parent
    metadata = args.metadata or (root / "article_metadata.csv")
    print(
        run_extract_batch(
            root=root,
            metadata_csv=metadata,
            model=args.model,
            seed=args.seed,
            top_k=args.top_k,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
