"""Load field queries and scoring rubrics from config/."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
QUERIES_PATH = CONFIG_DIR / "queries.json"
RUBRICS_PATH = CONFIG_DIR / "rubrics.md"
FIELD_CONFIG = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))


def load_rubric() -> str:
    return RUBRICS_PATH.read_text(encoding="utf-8").strip()
