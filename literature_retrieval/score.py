"""
Score extracted study facts with an LLM rubric (0-5).

Standalone (test with existing CSV):
  python score.py --csv ../pmc_articles/article_information.csv
  # -> ../pmc_articles/article_score.csv

Called from extract.py after extraction finishes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ollama_client import OllamaClient, OllamaConfig
from config import load_rubric

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
        raise ValueError("empty model response (no JSON to parse)")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))

    scores: Dict[str, int] = {}
    for key in SCORE_COLUMNS:
        value = int(data.get(key, 0))
        scores[key] = max(0, min(5, value))
    return scores


def _preview(text: str, limit: int = 400) -> str:
    text = (text or "").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text)} chars total)"


def _format_score_failure(err: Exception, raw: Optional[Dict[str, Any]]) -> str:
    """Human-readable reason for a scoring failure."""
    lines = [f"score failed: {type(err).__name__}: {err}"]
    if raw:
        response = raw.get("response") or ""
        thinking = raw.get("thinking") or ""
        lines.append(
            "  ollama meta: done="
            f"{raw.get('done')} done_reason={raw.get('done_reason')}"
            f" eval_count={raw.get('eval_count')}"
            f" prompt_eval_count={raw.get('prompt_eval_count')}"
            f" response_chars={len(response)}"
            f" thinking_chars={len(thinking)}"
        )
        if thinking and not response:
            lines.append(
                "  hint: model returned thinking but empty response "
                "(token budget may have been spent on reasoning)"
            )
        if response:
            lines.append(f"  response preview: {_preview(response)}")
        if thinking:
            lines.append(f"  thinking preview: {_preview(thinking)}")
    return "\n".join(lines)


def score_paper(row: Dict[str, Any], client: OllamaClient) -> Dict[str, int]:
    prompt = build_prompt(row)
    raw: Optional[Dict[str, Any]] = None
    try:
        raw = client.generate_raw(prompt, system=SYSTEM_PROMPT)
        response = raw.get("response") or ""
        return _parse_scores(response)
    except Exception as e:
        raise RuntimeError(_format_score_failure(e, raw)) from e


def make_score_client(
    model: str = "qwen3.5:9b",
    seed: int = 42,
    thinking: bool | str = False,
    num_predict: int = 8192,
    num_ctx: int = 32768,
) -> OllamaClient:
    """Deterministic scoring client with enough context for thinking + JSON."""
    return OllamaClient(
        OllamaConfig(
            model=model,
            thinking=thinking,
            temperature=0.0,
            options={
                "seed": seed,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        )
    )


def score_dataframe(df: pd.DataFrame, client: OllamaClient) -> pd.DataFrame:
    opts = client.config.options
    print(
        f"score settings: thinking={client.config.thinking!r}"
        f" num_predict={opts.get('num_predict')}"
        f" num_ctx={opts.get('num_ctx')}"
    )
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        item = row.to_dict()
        print(f"scoring Paper_ID={item.get('Paper_ID')} PMCID={item.get('PMCID')}")
        try:
            scores = score_paper(item, client)
        except Exception as e:
            # score_paper already wraps parse errors with ollama diagnostics
            print(f"  {e}")
            scores = {k: "" for k in SCORE_COLUMNS}
        rows.append(
            {
                "Paper_ID": item.get("Paper_ID", ""),
                "PMCID": item.get("PMCID", ""),
                **scores,
            }
        )
        print(f"  scores={scores}")
    return pd.DataFrame(rows, columns=["Paper_ID", "PMCID"] + SCORE_COLUMNS)


def run_scoring(
    input_path: Path,
    output_path: Optional[Path] = None,
    client: Optional[OllamaClient] = None,
    model: str = "qwen3.5:9b",
) -> pd.DataFrame:
    """Score rows from article_information.csv and save to article_score.csv."""
    client = client or make_score_client(model=model)
    if not client.is_available():
        raise RuntimeError(
            f"Ollama not available. Start with: ollama serve && ollama pull {model}"
        )

    root = Path(__file__).resolve().parent.parent / "pmc_articles"
    out = output_path or (root / "article_score.csv")
    df = pd.read_csv(input_path)
    scored = score_dataframe(df, client)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.is_file():
        prev = pd.read_csv(out)
        if "PMCID" in prev.columns:
            drop_ids = set(scored["PMCID"].astype(str))
            prev = prev[~prev["PMCID"].astype(str).isin(drop_ids)]
            scored = pd.concat([prev, scored], ignore_index=True)

    scored.to_csv(out, index=False)
    return scored


def score_extract_row(
    row: Dict[str, Any],
    score_csv: Path,
    client: Optional[OllamaClient] = None,
    model: str = "qwen3.5:9b",
) -> Dict[str, Any]:
    """Score one extract row and upsert into article_score.csv."""
    client = client or make_score_client(model=model)
    opts = client.config.options
    print(
        f"score settings: thinking={client.config.thinking!r}"
        f" num_predict={opts.get('num_predict')}"
        f" num_ctx={opts.get('num_ctx')}"
    )
    scores = score_paper(row, client)
    result = {
        "Paper_ID": row.get("Paper_ID", ""),
        "PMCID": row.get("PMCID", ""),
        **scores,
    }

    if score_csv.is_file():
        df = pd.read_csv(score_csv)
    else:
        df = pd.DataFrame(columns=["Paper_ID", "PMCID"] + SCORE_COLUMNS)

    pmcid = str(result.get("PMCID") or "")
    if "PMCID" in df.columns and pmcid:
        df = df[df["PMCID"].astype(str) != pmcid]
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df = df[["Paper_ID", "PMCID"] + SCORE_COLUMNS]
    score_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(score_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score extracted article information with Ollama"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Input article_information.csv (default: ../pmc_articles/article_information.csv)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output score CSV (default: ../pmc_articles/article_score.csv)",
    )
    parser.add_argument("-m", "--model", default="qwen3.5:9b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thinking",
        choices=["false", "low", "medium", "high", "max"],
        default="false",
        help="Ollama thinking for scoring (default: false; qwen3.5 ignores low/high)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "pmc_articles"
    input_path = args.csv or (root / "article_information.csv")
    output_path = args.out or (root / "article_score.csv")
    if not input_path.is_file():
        raise SystemExit(f"CSV not found: {input_path}")

    thinking: bool | str = False if args.thinking == "false" else args.thinking
    client = make_score_client(
        model=args.model,
        seed=args.seed,
        thinking=thinking,
    )
    scored = run_scoring(input_path, output_path=output_path, client=client)
    print(f"saved scores -> {output_path} ({len(scored)} rows)")


if __name__ == "__main__":
    main()
