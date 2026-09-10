#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
OUT_DIR="${1:-./pmc_articles}"
MAX_RESULTS="${2:-20}"
YEAR="${3:-2020}"
QUERY='("mild cognitive impairment"[tiab] OR MCI[tiab]) AND ("Alzheimer disease"[tiab] OR "Alzheimer'\''s disease"[tiab]) AND (plasma[tiab] OR serum[tiab] OR blood[tiab]) AND (proteomic*[tiab] OR "protein biomarker*"[tiab] OR "plasma protein*"[tiab] OR "serum protein*"[tiab])'
python3 article_client.py "$QUERY" -o "$OUT_DIR" -n "$MAX_RESULTS" -y "$YEAR"
