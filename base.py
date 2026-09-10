"""
Shared base classes for external data sources (PubMed, Google Patents, etc.).

Provides a unified article dataclass and abstract client interface so all
external sources produce the same shape of data for the ingestion pipeline.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExternalArticle:
    """Unified article/patent record from any external source.

    Fields map to the required metadata for vector DB storage:
      DB, year, DOI, PMID/PMCID, title, journal, author, source_url
    """

    source_id: str  # PMID, patent number, etc.
    source: str  # "pubmed", "pmc", "google_patents", ...
    title: str
    abstract: str
    authors: List[str]  # Author names or assignees
    year: str
    link: str  # DOI URL, PubMed URL, patents.google.com URL, etc.
    journal: str  # Journal name, or "US Patent" / "EP Patent" for patents
    language: str = ""  # SerpApi language code, e.g. "en", "zh", "ja"
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # Source-specific fields

    def to_metadata(self, query: str) -> Dict[str, str]:
        """Flatten to a metadata dict suitable for vector DB storage.

        ChromaDB / Qdrant require flat string/int/float values.
        """
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
        """Search the source and return a list of record IDs."""

    @abstractmethod
    def fetch_details(self, ids: List[str]) -> List[ExternalArticle]:
        """Fetch full details for a list of record IDs."""

    def search_and_fetch(self, query: str, max_results: int = 20) -> List[ExternalArticle]:
        """Convenience: search then fetch in one call."""
        ids = self.search(query, max_results)
        if not ids:
            logger.info("No results found for query: %s", query)
            return []
        logger.info("Found %d IDs, fetching details...", len(ids))
        return self.fetch_details(ids)

    def download_pdf(self, article: ExternalArticle, output_dir: Path) -> Optional[Path]:
        """Download a PDF for the article if available. Returns path or None.

        Subclasses should override this if they support PDF downloads.
        """
        return None
