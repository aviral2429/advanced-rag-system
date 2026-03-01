"""
loader.py — Multi-PDF document loader with rich metadata extraction.

Improvement over baseline:
- Layout-aware extraction via pdfplumber (handles columns, tables).
- pypdf fallback for malformed / edge-case PDFs.
- Per-file SHA-256 hash for change detection and cache invalidation.
- Scanned-PDF detection with user warning.
- Structured Document dataclass with full provenance metadata.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    A single page extracted from a PDF file.

    Attributes
    ----------
    content:
        Raw text content of the page.
    metadata:
        Provenance information (file, page numbers, hash, timestamps).
    """

    content: str
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #

    @property
    def source(self) -> str:
        """Short human-readable source label for citations."""
        filename = self.metadata.get("filename", "unknown")
        page = self.metadata.get("page_number", "?")
        return f"{filename}, p.{page}"

    @property
    def doc_id(self) -> str:
        """Unique identifier combining file hash and page number."""
        sha = self.metadata.get("sha256", "")[:8]
        page = self.metadata.get("page_number", 0)
        return f"{sha}-p{page}"

    def is_empty(self) -> bool:
        return len(self.content.strip()) < 20


class PDFLoader:
    """
    Loads one or many PDF files and returns a flat list of Document objects
    (one per page).

    Parameters
    ----------
    min_chars_per_page:
        Pages with fewer characters are flagged as potentially scanned.
    """

    def __init__(self, min_chars_per_page: int = 50) -> None:
        self.min_chars_per_page = min_chars_per_page

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_directory(self, path: str | Path) -> List[Document]:
        """
        Recursively load all PDF files found under *path*.

        Returns a flat list of Document objects (one per page, all files).
        """
        base = Path(path)
        if not base.exists():
            raise FileNotFoundError(f"PDF directory not found: {base}")

        pdf_paths = sorted(base.rglob("*.pdf"))
        if not pdf_paths:
            warnings.warn(f"No PDF files found in {base}", stacklevel=2)
            return []

        documents: List[Document] = []
        for pdf_path in pdf_paths:
            try:
                docs = self.load_file(pdf_path)
                documents.extend(docs)
                logger.info("Loaded %d pages from %s", len(docs), pdf_path.name)
            except Exception as exc:
                logger.error("Failed to load %s: %s", pdf_path.name, exc)

        logger.info("Total pages loaded: %d from %d files", len(documents), len(pdf_paths))
        return documents

    def load_file(self, path: str | Path) -> List[Document]:
        """
        Load a single PDF file.

        Tries pdfplumber first; falls back to pypdf on failure.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        sha256 = self._compute_sha256(path)
        file_size = path.stat().st_size
        extraction_ts = datetime.now(tz=timezone.utc).isoformat()

        # Try primary parser
        try:
            documents = self._extract_with_pdfplumber(path, sha256, file_size, extraction_ts)
        except Exception as exc:
            logger.warning("pdfplumber failed for %s (%s); trying pypdf.", path.name, exc)
            documents = self._extract_with_pypdf(path, sha256, file_size, extraction_ts)

        # Warn if document appears to be scanned
        scanned_pages = [d for d in documents if self._detect_scanned(d.content)]
        if scanned_pages:
            warnings.warn(
                f"{path.name}: {len(scanned_pages)} page(s) appear to be scanned images. "
                "Consider running OCR (e.g. pytesseract) before indexing.",
                stacklevel=2,
            )

        return [d for d in documents if not d.is_empty()]

    # ------------------------------------------------------------------ #
    # Internal extraction methods
    # ------------------------------------------------------------------ #

    def _extract_with_pdfplumber(
        self,
        path: Path,
        sha256: str,
        file_size: int,
        extraction_ts: str,
    ) -> List[Document]:
        """Extract text using pdfplumber (layout-aware, handles tables/columns)."""
        import pdfplumber  # lazy import — not needed at module load

        documents: List[Document] = []
        with pdfplumber.open(str(path)) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                doc = Document(
                    content=text,
                    metadata={
                        "filename": path.name,
                        "file_path": str(path.resolve()),
                        "page_number": page_num,
                        "total_pages": total_pages,
                        "file_size_bytes": file_size,
                        "sha256": sha256,
                        "extraction_ts": extraction_ts,
                        "parser": "pdfplumber",
                    },
                )
                documents.append(doc)
        return documents

    def _extract_with_pypdf(
        self,
        path: Path,
        sha256: str,
        file_size: int,
        extraction_ts: str,
    ) -> List[Document]:
        """Extract text using pypdf (pure-Python fallback)."""
        from pypdf import PdfReader  # lazy import

        documents: List[Document] = []
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            doc = Document(
                content=text,
                metadata={
                    "filename": path.name,
                    "file_path": str(path.resolve()),
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "file_size_bytes": file_size,
                    "sha256": sha256,
                    "extraction_ts": extraction_ts,
                    "parser": "pypdf",
                },
            )
            documents.append(doc)
        return documents

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA-256 hash of a file for change-detection / cache keying."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _detect_scanned(self, text: str) -> bool:
        """
        Heuristic: if a page yields fewer than min_chars_per_page printable
        characters, it is likely a scanned image without embedded text.
        """
        return len(text.strip()) < self.min_chars_per_page
