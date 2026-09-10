"""Shared models, IO helpers, Ollama client, and field/rubric config."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import requests

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
ROOT = PROJECT_ROOT  # backward-compatible alias for project root
CONFIGS_DIR = PACKAGE_ROOT / "configs"
QUERIES_PATH = CONFIGS_DIR / "queries.json"
RUBRICS_PATH = CONFIGS_DIR / "rubrics.md"
FIELD_CONFIG = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))


def load_rubric() -> str:
    return RUBRICS_PATH.read_text(encoding="utf-8").strip()


# --- models (from base.py) ---


@dataclass
class ExternalArticle:
    """Unified article/patent record from any external source."""

    source_id: str
    source: str
    title: str
    abstract: str
    authors: List[str]
    year: str
    link: str
    journal: str
    language: str = ""
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self, query: str) -> Dict[str, str]:
        pmid = self.source_id or ""
        pmcid = self.pmcid or ""
        if pmid and pmcid:
            id_value = f"{pmid}/{pmcid}"
        else:
            id_value = pmid or pmcid
        return {
            "DB": self.source,
            "year": self.year or "",
            "DOI": self.doi or "",
            "PMID/PMCID": id_value,
            "title": self.title or "",
            "journal": self.journal or "",
            "author": self.authors[0] if self.authors else "",
            "source_url": self.link or "",
        }


class BaseSourceClient(ABC):
    """Abstract interface for external data source clients."""

    @abstractmethod
    def search(self, query: str, max_results: int = 20) -> List[str]:
        ...

    @abstractmethod
    def fetch_details(self, ids: List[str]) -> List[ExternalArticle]:
        ...

    def search_and_fetch(self, query: str, max_results: int = 20) -> List[ExternalArticle]:
        ids = self.search(query, max_results)
        if not ids:
            logger.info("No results found for query: %s", query)
            return []
        logger.info("Found %d IDs, fetching details...", len(ids))
        return self.fetch_details(ids)

    def download_pdf(self, article: ExternalArticle, output_dir: Path) -> Optional[Path]:
        return None


# --- IO helpers (from pipeline_io.py) ---


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit(repo: Path = PROJECT_ROOT) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def source_hashes(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        if not p.is_file():
            continue
        try:
            key = str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            key = str(p)
        out[key] = sha256_file(p)
    return out


def load_paper_id_map(metadata_csv: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not metadata_csv.is_file():
        return mapping
    with metadata_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pmcid = normalize_pmcid(str(row.get("PMID/PMCID") or ""))
            paper_id = str(row.get("Paper_ID") or "").strip()
            if pmcid and paper_id:
                mapping[pmcid] = paper_id
    return mapping


def normalize_pmcid(value: str) -> str:
    parts = [p.strip() for p in str(value).split("/") if p.strip()]
    token = next((p for p in parts if p.upper().startswith("PMC")), parts[-1] if parts else "")
    token = token.upper().replace("PMC", "").strip()
    return f"PMC{token}" if token else ""


def assign_paper_ids(
    pmcids: List[str],
    existing: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    existing = dict(existing or {})
    used = set(existing.values())
    next_n = 1
    for pmcid in sorted(set(pmcids)):
        if pmcid in existing:
            continue
        while f"P{next_n:03d}" in used:
            next_n += 1
        pid = f"P{next_n:03d}"
        existing[pmcid] = pid
        used.add(pid)
        next_n += 1
    return existing


def write_csv_atomic(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def env_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for pkg in ("pandas", "numpy", "requests", "sentence_transformers"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "missing"
    return versions


def find_xml_only_dirs(out_dir: Path) -> List[str]:
    uids: List[str] = []
    for article_dir in sorted(Path(out_dir).glob("PMC_UID_*")):
        if (article_dir / "article.xml").is_file() and not (
            article_dir / "parsed_sections.json"
        ).is_file():
            uids.append(article_dir.name.replace("PMC_UID_", "", 1))
    return uids


# --- Ollama client ---


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"
    thinking: Union[bool, str] = False
    temperature: float = 0.3
    top_p: float = 0.9
    timeout: int = 600
    options: Dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.session = requests.Session()

    def is_available(self) -> bool:
        try:
            resp = self.session.get(f"{self.config.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self) -> List[str]:
        try:
            resp = self.session.get(f"{self.config.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error("Failed to list models: %s", e)
            return []

    def version(self) -> str:
        try:
            resp = self.session.get(f"{self.config.base_url}/api/version", timeout=5)
            if resp.status_code == 200:
                return str(resp.json().get("version") or "")
        except Exception:
            pass
        return ""

    def model_digest(self, model: Optional[str] = None) -> str:
        name = model or self.config.model
        try:
            resp = self.session.post(
                f"{self.config.base_url}/api/show",
                json={"name": name},
                timeout=30,
            )
            if resp.status_code != 200:
                return ""
            data = resp.json()
            details = data.get("details") or {}
            return str(
                data.get("modelfile_hash")
                or data.get("digest")
                or details.get("parent_model")
                or data.get("model_info", {}).get("general.basename")
                or ""
            )
        except Exception:
            return ""

    def provenance(self, model: Optional[str] = None) -> Dict[str, str]:
        name = model or self.config.model
        return {
            "base_url": self.config.base_url,
            "model": name,
            "ollama_version": self.version(),
            "model_digest": self.model_digest(name),
        }

    def generate_raw(self, prompt: str, system: str = None) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "think": self.config.thinking,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                **self.config.options,
            },
        }
        if system:
            payload["system"] = system
        try:
            logger.info("Generating with model=%s", self.config.model)
            response = self.session.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.config.timeout}s. "
                "Try increasing timeout or using a smaller model."
            )
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def generate(self, prompt: str, system: str = None) -> str:
        result = self.generate_raw(prompt, system=system)
        return result.get("response", "") or ""

    def chat(self, messages: List[Dict[str, str]], system: str = None) -> str:
        url = f"{self.config.base_url}/api/chat"
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        payload = {
            "model": self.config.model,
            "messages": chat_messages,
            "stream": False,
            "think": self.config.thinking,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                **self.config.options,
            },
        }
        try:
            logger.info("Chat with model=%s, %d messages", self.config.model, len(messages))
            response = self.session.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama chat timed out after {self.config.timeout}s. "
                "Try increasing timeout or using a smaller model."
            )
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama chat API error: {e}")
