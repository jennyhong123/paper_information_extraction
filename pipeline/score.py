"""Score extracted study facts with an LLM rubric (0-5)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import (
    RUBRICS_PATH,
    OllamaClient,
    OllamaConfig,
    append_jsonl,
    atomic_write_json,
    load_rubric,
    sha256_file,
    sha256_text,
    utc_now,
    write_csv_atomic,
)

SCORE_COLUMNS = [
    "MCI_score",
    "AD_score",
    "Blood_score",
    "Protein_score",
    "Quality_score",
    "Valid_score",
]

SYSTEM_PROMPT = (
    "You are a careful biomedical study scorer.\n"
    "Use ONLY the extracted information provided.\n"
    "Return JSON only. No markdown fences. No explanation."
)


def build_prompt(row: Dict[str, Any]) -> str:
    rubric = load_rubric()
    return (
        "You are evaluating a biomedical biomarker study.\n\n"
        "Use ONLY the extracted information below.\n"
        "Do not assume information that is not explicitly provided.\n\n"
        f"{rubric}\n\n"
        "[Extracted Study Information]\n\n"
        f"Cohort:\n{row.get('cohort', '')}\n\n"
        f"Sample size:\n{row.get('sample_size', '')}\n\n"
        f"Comparison group:\n{row.get('comparison_group', '')}\n\n"
        f"Sample type:\n{row.get('sample_type', '')}\n\n"
        f"Experimental technique:\n{row.get('experimental_technique', '')}\n\n"
        f"Biomarker:\n{row.get('biomarker', '')}\n\n"
        f"Key result:\n{row.get('key_result', '')}\n\n"
        "Return JSON only in this exact schema:\n\n"
        "{\n"
        '    "MCI_score": 0,\n'
        '    "AD_score": 0,\n'
        '    "Blood_score": 0,\n'
        '    "Protein_score": 0,\n'
        '    "Quality_score": 0,\n'
        '    "Valid_score": 0\n'
        "}\n\n"
        "All scores must be integers from 0 to 5.\n"
    )


def _parse_scores(response: str) -> Dict[str, int]:
    text = (response or "").replace("```json", "").replace("```", "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    scores: Dict[str, int] = {}
    for key in SCORE_COLUMNS:
        if key not in data:
            raise ValueError(f"missing score key: {key}")
        value = int(data[key])
        if value < 0 or value > 5:
            raise ValueError(f"{key} out of range: {value}")
        scores[key] = value
    return scores


def score_fingerprint(extraction_path: Path, model: str, seed: int, temperature: float) -> str:
    payload = {
        "extraction_sha256": sha256_file(extraction_path),
        "rubric_sha256": sha256_file(RUBRICS_PATH),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "model": model,
        "seed": seed,
        "temperature": temperature,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def make_score_client(
    model: str = "qwen3.5:9b",
    seed: int = 42,
    thinking: bool | str = False,
    temperature: float = 0.0,
    num_predict: int = 8192,
    num_ctx: int = 32768,
) -> OllamaClient:
    return OllamaClient(
        OllamaConfig(
            model=model,
            thinking=thinking,
            temperature=temperature,
            options={"seed": seed, "num_predict": num_predict, "num_ctx": num_ctx},
        )
    )


def score_paper(row: Dict[str, Any], client: OllamaClient) -> Dict[str, Any]:
    prompt = build_prompt(row)
    raw = client.generate_raw(prompt, system=SYSTEM_PROMPT)
    response = raw.get("response") or ""
    scores = _parse_scores(response)
    return {"scores": scores, "raw_response": response, "ollama": {
        "done": raw.get("done"),
        "done_reason": raw.get("done_reason"),
        "eval_count": raw.get("eval_count"),
    }}


def run_score_batch(
    root: Path,
    model: str = "qwen3.5:9b",
    seed: int = 42,
    temperature: float = 0.0,
    thinking: bool | str = False,
    force: bool = False,
    dry_run: bool = False,
    max_retries: int = 2,
    status_log: Optional[Path] = None,
    client: Optional[OllamaClient] = None,
) -> Dict[str, Any]:
    root = Path(root)
    paths = sorted(root.glob("PMC_UID_*/extraction.json"))
    stats = {"ok": 0, "skipped": 0, "failed": 0, "needs_work": 0}
    if not dry_run:
        client = client or make_score_client(
            model=model, seed=seed, thinking=thinking, temperature=temperature
        )
        if not client.is_available():
            raise RuntimeError(f"Ollama not available for model={model}")
        model_digest = client.model_digest(model)
    else:
        model_digest = ""

    for ext_path in paths:
        data = json.loads(ext_path.read_text(encoding="utf-8"))
        if data.get("status") != "ok":
            continue
        out_path = ext_path.parent / "score.json"
        fp = score_fingerprint(ext_path, model, seed, temperature)
        if out_path.is_file() and not force:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("status") == "ok" and prev.get("fingerprint") == fp:
                stats["skipped"] += 1
                continue
        if dry_run:
            stats["needs_work"] += 1
            continue

        row = data.get("answers") or {}
        pmcid = row.get("PMCID") or data.get("pmcid") or ""
        paper_id = row.get("Paper_ID") or data.get("paper_id") or ""
        last_err = ""
        for _ in range(max_retries):
            try:
                result = score_paper(row, client)
                payload = {
                    "status": "ok",
                    "fingerprint": fp,
                    "pmcid": pmcid,
                    "paper_id": paper_id,
                    "extraction_sha256": sha256_file(ext_path),
                    "model": model,
                    "model_digest": model_digest,
                    "seed": seed,
                    "temperature": temperature,
                    "ts": utc_now(),
                    **result,
                }
                atomic_write_json(out_path, payload)
                stats["ok"] += 1
                if status_log:
                    append_jsonl(
                        status_log,
                        {"ts": utc_now(), "stage": "score", "pmcid": pmcid, "status": "ok"},
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
                        "stage": "score",
                        "pmcid": pmcid,
                        "status": "failed",
                        "error": last_err,
                    },
                )

    if not dry_run:
        aggregate_score_csv(root, root / "article_score.csv")
    return stats


def aggregate_score_csv(root: Path, csv_path: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(root).glob("PMC_UID_*/score.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "ok":
            continue
        scores = data.get("scores") or {}
        rows.append(
            {
                "Paper_ID": data.get("paper_id", ""),
                "PMCID": data.get("pmcid", ""),
                **{k: scores.get(k, "") for k in SCORE_COLUMNS},
            }
        )
    rows.sort(key=lambda r: str(r.get("PMCID") or ""))
    write_csv_atomic(csv_path, ["Paper_ID", "PMCID"] + SCORE_COLUMNS, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score extracted articles")
    parser.add_argument("root", type=Path, nargs="?", default=None)
    parser.add_argument("-m", "--model", default="qwen3.5:9b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root or (Path(__file__).resolve().parent.parent / "pmc_articles")
    print(run_score_batch(root, model=args.model, seed=args.seed, force=args.force))


if __name__ == "__main__":
    main()
