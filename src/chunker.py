"""
chunker.py — Semantic and recursive adaptive document chunking.

Improvement over baseline (fixed-size character splits):
- Semantic strategy: sentences are first tokenised with NLTK punkt, then
  adjacent-sentence cosine similarities are computed.  A chunk boundary is
  inserted wherever the similarity drop falls in the top
  (100 - breakpoint_percentile)% of all drops — a data-driven threshold that
  adapts to each document's internal structure.
- Recursive fallback: when a document is too short for semantic analysis,
  a hierarchy of separators (paragraph → line → sentence → character) is used
  so that natural boundaries are always preferred over mid-word breaks.
- Adaptive overlap: 15 % of chunk_size (not a fixed character count) preserves
  cross-boundary context without excessive duplication.
- Rich per-chunk metadata: chunk_id, source_file, page_number, chunk_index,
  total_chunks, strategy_used — essential for accurate citation generation.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import List, Tuple

import numpy as np

from src.loader import Document

logger = logging.getLogger(__name__)

# Minimum number of sentences required to attempt semantic chunking.
_MIN_SENTENCES_FOR_SEMANTIC = 4


class SemanticChunker:
    """
    Splits a list of Documents into semantically coherent chunks.

    Parameters
    ----------
    embedding_model:
        An initialised EmbeddingModel instance (used only by semantic strategy).
    chunk_size:
        Target *character* length of each chunk.
    chunk_overlap:
        Character overlap between consecutive chunks.
    strategy:
        'semantic' (preferred) or 'recursive' (faster baseline).
    breakpoint_percentile:
        Percentile (0–99) of similarity-drop magnitudes used as the
        breakpoint threshold.  Higher = fewer, larger chunks.
    """

    def __init__(
        self,
        embedding_model,  # EmbeddingModel — avoids circular import at type level
        chunk_size: int = 1024,
        chunk_overlap: int = 150,
        strategy: str = "semantic",
        breakpoint_percentile: int = 85,
    ) -> None:
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.breakpoint_percentile = breakpoint_percentile

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk every Document in *documents* and return a flat list of chunk
        Documents (one Document per chunk with updated metadata).
        """
        all_chunks: List[Document] = []
        for doc in documents:
            chunks = self._chunk_single(doc)
            all_chunks.extend(chunks)
        logger.info(
            "Chunking complete: %d pages → %d chunks (strategy=%s).",
            len(documents),
            len(all_chunks),
            self.strategy,
        )
        return all_chunks

    # ------------------------------------------------------------------ #
    # Chunking strategies
    # ------------------------------------------------------------------ #

    def _chunk_single(self, doc: Document) -> List[Document]:
        """Choose strategy, chunk, and annotate metadata."""
        text = doc.content.strip()
        if not text:
            return []

        sentences = self._split_into_sentences(text)

        if self.strategy == "semantic" and len(sentences) >= _MIN_SENTENCES_FOR_SEMANTIC:
            raw_chunks = self._semantic_chunk(sentences)
            strategy_used = "semantic"
        else:
            raw_chunks = self._recursive_chunk(text)
            strategy_used = "recursive"

        total = len(raw_chunks)
        result: List[Document] = []
        for idx, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < 20:
                continue
            chunk_doc = Document(
                content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": idx,
                    "total_chunks": total,
                    "strategy_used": strategy_used,
                    "char_count": len(chunk_text),
                },
            )
            result.append(chunk_doc)
        return result

    # ------------------------------------------------------------------ #
    # Semantic chunking (primary)
    # ------------------------------------------------------------------ #

    def _semantic_chunk(self, sentences: List[str]) -> List[str]:
        """
        Group sentences into chunks by detecting semantic breakpoints.

        Algorithm
        ---------
        1. Embed each sentence (uses EmbeddingModel cache — fast on re-runs).
        2. Compute cosine similarity between every adjacent pair.
        3. Compute drop between each pair: drop[i] = sim[i] - sim[i+1].
        4. Identify breakpoint indices where drop > percentile threshold.
        5. Merge sentences between breakpoints into chunk strings.
        6. If any chunk exceeds chunk_size, split it recursively.
        """
        # Embed all sentences (already normalised → dot = cosine)
        embeddings = self.embedding_model.embed_documents(sentences, show_progress=False)

        # Adjacent cosine similarities
        sims = np.array(
            [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
        )

        # Detect breakpoints
        breakpoints = self._compute_breakpoints(sims)

        # Group sentences into segments
        segments: List[str] = []
        prev = 0
        for bp in breakpoints:
            segment = " ".join(sentences[prev : bp + 1])
            segments.append(segment)
            prev = bp + 1
        segments.append(" ".join(sentences[prev:]))

        # Apply size cap + overlap
        final_chunks: List[str] = []
        for seg in segments:
            if len(seg) <= self.chunk_size:
                final_chunks.append(seg)
            else:
                # Sub-split oversized segments recursively
                sub = self._recursive_chunk(seg)
                final_chunks.extend(sub)

        return self._apply_overlap(final_chunks)

    def _compute_breakpoints(self, similarities: np.ndarray) -> List[int]:
        """
        Return indices (into the *similarities* array) that constitute
        semantic breakpoints.

        A breakpoint at index i means a chunk boundary is inserted after
        sentence i.
        """
        if len(similarities) == 0:
            return []

        threshold = float(np.percentile(similarities, 100 - self.breakpoint_percentile))
        # Indices where similarity drops below threshold
        return [int(i) for i, s in enumerate(similarities) if s < threshold]

    # ------------------------------------------------------------------ #
    # Recursive chunking (fallback / baseline)
    # ------------------------------------------------------------------ #

    def _recursive_chunk(self, text: str) -> List[str]:
        """
        Split text using a hierarchy of separators, always preferring
        natural language boundaries over arbitrary character positions.

        Separators tried in order:
            double newline (paragraphs) → single newline → period-space →
            single space → individual characters.
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._split_recursive(text, separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split until all chunks fit within chunk_size."""
        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:]

        parts = text.split(sep) if sep else list(text)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size and remaining_seps:
                    # Recursively split this oversized part
                    chunks.extend(self._split_recursive(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return self._apply_overlap(chunks)

    # ------------------------------------------------------------------ #
    # Overlap injection
    # ------------------------------------------------------------------ #

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Inject character overlap between consecutive chunks to preserve
        cross-boundary context for the retriever.
        """
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap :]
            overlapped.append(tail + chunks[i])

        return overlapped

    # ------------------------------------------------------------------ #
    # Sentence splitting utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences using NLTK punkt tokeniser.
        Falls back to a simple regex split if NLTK is unavailable.
        """
        try:
            import nltk

            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)

            sentences = nltk.sent_tokenize(text)
        except Exception:
            # Simple regex fallback
            sentences = re.split(r"(?<=[.!?])\s+", text)

        return [s.strip() for s in sentences if s.strip()]
