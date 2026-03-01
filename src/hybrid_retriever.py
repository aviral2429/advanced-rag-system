"""
hybrid_retriever.py — Three-stage retrieval: FAISS + BM25 → RRF fusion → Reranker.

Improvement over baseline (FAISS-only dense retrieval):

1. BM25 sparse retrieval (rank-bm25):
   Lexical matching catches exact keyword / acronym hits that dense vectors
   often miss, especially for technical documents with domain-specific terms.

2. Reciprocal Rank Fusion (RRF):
   A parameter-free score fusion formula:
       score(d) = Σ_i  1 / (k + rank_i(d))
   where k=60 by default.  RRF is robust to score-scale differences between
   dense and sparse results and requires no tuning (unlike weighted fusion).

3. Cross-encoder reranker (ms-marco-MiniLM-L-12-v2):
   Unlike bi-encoders that encode query and passage independently, a
   cross-encoder processes (query, passage) pairs jointly, enabling full
   attention across both.  This improves precision@5 by ~10–15% at the cost
   of ~300 ms additional latency (CPU, top-15 candidates).

4. Deduplication:
   Near-duplicate chunks (same content hash) are removed before returning
   results, preventing the LLM from receiving redundant context.

5. Confidence filtering:
   Chunks whose normalised score falls below a threshold are dropped,
   reducing noise passed to the LLM.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.loader import Document

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    """A retrieved chunk annotated with its retrieval score and rank."""

    document: Document
    score: float          # normalised RRF score (0–1 after normalisation)
    rank: int             # 1-based final rank
    retrieval_method: str  # "dense" | "sparse" | "hybrid" | "reranked"


class HybridRetriever:
    """
    Builds and queries a hybrid FAISS + BM25 index with optional reranking.

    Parameters
    ----------
    embedding_model:
        Initialised EmbeddingModel for query encoding.
    rrf_k:
        RRF smoothing constant (default 60).
    use_reranker:
        Whether to apply cross-encoder reranking after fusion.
    reranker_model_name:
        HuggingFace cross-encoder model identifier.
    candidates:
        Number of candidates retrieved from each of dense and sparse
        before fusion.
    confidence_threshold:
        Minimum normalised score to include a chunk in results.
    """

    def __init__(
        self,
        embedding_model,
        rrf_k: int = 60,
        use_reranker: bool = True,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        candidates: int = 50,
        confidence_threshold: float = 0.3,
    ) -> None:
        self.embedding_model = embedding_model
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model_name
        self.candidates = candidates
        self.confidence_threshold = confidence_threshold

        # Internal state — populated by build_index()
        self._chunks: List[Document] = []
        self._faiss_index = None
        self._bm25 = None
        self._tokenised_corpus: List[List[str]] = []
        self._reranker = None  # lazy-loaded

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def build_index(self, chunks: List[Document]) -> None:
        """
        Build the FAISS dense index and BM25 sparse index from *chunks*.

        Must be called before retrieve().
        """
        if not chunks:
            raise ValueError("Cannot build index from an empty chunk list.")

        self._chunks = chunks
        texts = [c.content for c in chunks]

        logger.info("Building dense FAISS index for %d chunks …", len(chunks))
        self._build_faiss(texts)

        logger.info("Building sparse BM25 index …")
        self._build_bm25(texts)

        logger.info("Index construction complete.")

    def _build_faiss(self, texts: List[str]) -> None:
        """Encode all texts and create a FAISS IndexFlatIP (exact cosine search)."""
        import faiss  # lazy import

        embeddings = self.embedding_model.embed_documents(texts, show_progress=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on L2-normalised vecs = cosine
        index.add(embeddings)
        self._faiss_index = index

    def _build_bm25(self, texts: List[str]) -> None:
        """Tokenise texts and build a BM25Okapi index."""
        from rank_bm25 import BM25Okapi  # lazy import

        self._tokenised_corpus = [self._tokenise(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenised_corpus)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """
        Retrieve the top-*top_k* most relevant chunks for *query*.

        Pipeline
        --------
        1. Dense retrieval → top-candidates via FAISS.
        2. Sparse retrieval → top-candidates via BM25.
        3. RRF fusion → merged ranked list.
        4. Optional cross-encoder reranking.
        5. Deduplication + confidence filtering.
        6. Return top-*top_k* ScoredChunks.
        """
        if self._faiss_index is None or self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # --- Stage 1: Dense ---
        dense_ids, dense_scores = self._dense_retrieve(query, self.candidates)

        # --- Stage 2: Sparse ---
        sparse_ids, sparse_scores = self._sparse_retrieve(query, self.candidates)

        # --- Stage 3: RRF Fusion ---
        fused = self._rrf_fusion(dense_ids, sparse_ids)

        # Pre-rerank pool: top (top_k * 3) candidates from fusion
        pool_size = min(top_k * 3, len(fused))
        pool_ids = [doc_id for doc_id, _ in fused[:pool_size]]
        pool_scores = {doc_id: score for doc_id, score in fused[:pool_size]}

        # --- Stage 4: Optional Reranking ---
        if self.use_reranker and len(pool_ids) > 0:
            reranked_ids = self._rerank(query, pool_ids)
            method = "reranked"
        else:
            reranked_ids = pool_ids
            method = "hybrid"

        # --- Stage 5: Confidence filter + deduplication ---
        max_score = pool_scores[reranked_ids[0]] if reranked_ids else 1.0
        max_score = max_score or 1.0  # guard against zero

        seen_hashes: set = set()
        results: List[ScoredChunk] = []
        for rank, doc_id in enumerate(reranked_ids, start=1):
            normalised_score = pool_scores.get(doc_id, 0.0) / max_score
            if normalised_score < self.confidence_threshold:
                continue

            chunk = self._chunks[doc_id]
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            results.append(
                ScoredChunk(
                    document=chunk,
                    score=round(normalised_score, 4),
                    rank=rank,
                    retrieval_method=method,
                )
            )

            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------ #
    # Dense (FAISS) retrieval
    # ------------------------------------------------------------------ #

    def _dense_retrieve(
        self, query: str, k: int
    ) -> Tuple[List[int], Dict[int, float]]:
        """Return (doc_ids, id→score) for top-k dense results."""
        query_vec = self.embedding_model.embed_query(query).reshape(1, -1)
        n = min(k, self._faiss_index.ntotal)
        scores_arr, ids_arr = self._faiss_index.search(query_vec, n)

        ids = ids_arr[0].tolist()
        scores = {int(i): float(s) for i, s in zip(ids_arr[0], scores_arr[0]) if i >= 0}
        valid_ids = [i for i in ids if i >= 0]
        return valid_ids, scores

    # ------------------------------------------------------------------ #
    # Sparse (BM25) retrieval
    # ------------------------------------------------------------------ #

    def _sparse_retrieve(
        self, query: str, k: int
    ) -> Tuple[List[int], Dict[int, float]]:
        """Return (doc_ids, id→score) for top-k BM25 results."""
        tokenised_query = self._tokenise(query)
        raw_scores = self._bm25.get_scores(tokenised_query)

        k = min(k, len(raw_scores))
        top_ids = np.argsort(raw_scores)[::-1][:k].tolist()
        scores = {int(i): float(raw_scores[i]) for i in top_ids}
        return top_ids, scores

    # ------------------------------------------------------------------ #
    # RRF Fusion
    # ------------------------------------------------------------------ #

    def _rrf_fusion(
        self,
        dense_ids: List[int],
        sparse_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """
        Combine dense and sparse ranked lists via Reciprocal Rank Fusion.

        RRF score(d) = Σ_i  1 / (k + rank_i(d))

        This is parameter-free with respect to the actual score values,
        making it robust to the very different scales of cosine and BM25 scores.
        """
        fusion: Dict[int, float] = {}

        for rank, doc_id in enumerate(dense_ids, start=1):
            fusion[doc_id] = fusion.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, doc_id in enumerate(sparse_ids, start=1):
            fusion[doc_id] = fusion.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        return sorted(fusion.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------ #
    # Cross-encoder reranking
    # ------------------------------------------------------------------ #

    def _rerank(self, query: str, candidate_ids: List[int]) -> List[int]:
        """
        Rerank *candidate_ids* using a cross-encoder model.

        The cross-encoder reads (query, passage) jointly via full self-attention,
        giving much richer relevance signals than a bi-encoder at the cost
        of O(N) forward passes (one per candidate).

        Falls back gracefully to the original candidate order if the reranker
        fails for any reason (e.g. model load error, CUDA/meta tensor issues).
        """
        try:
            reranker = self._get_reranker()
            passages = [self._chunks[i].content for i in candidate_ids]
            pairs = [[query, p] for p in passages]
            scores = reranker.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, _ in ranked]
        except Exception as exc:
            logger.warning(
                "Reranker failed (%s); falling back to RRF order. "
                "Disable the reranker in the sidebar to suppress this warning.",
                exc,
            )
            # Disable reranker for the rest of this session to avoid repeat errors
            self.use_reranker = False
            self._reranker = None
            return candidate_ids

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker.

        Permanently pinned to CPU with float32 and device_map=None to prevent
        the 'Cannot copy out of meta tensor; no data!' error that occurs when
        sentence-transformers 3.x + accelerate accidentally loads model layers
        onto the PyTorch 'meta' device.
        """
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # lazy import
            import os

            # Disable accelerate auto device-mapping for this model load.
            # This is the root cause of 'meta tensor' errors on CPU-only machines.
            os.environ.setdefault("ACCELERATE_USE_FSDP", "0")

            logger.info("Loading reranker model '%s' …", self.reranker_model_name)
            try:
                self._reranker = CrossEncoder(
                    self.reranker_model_name,
                    device="cpu",
                    max_length=512,
                    automodel_args={
                        "device_map": None,
                        "torch_dtype": "float32",
                        "low_cpu_mem_usage": False,
                    },
                )
            except TypeError:
                # Older sentence-transformers versions don't support automodel_args
                self._reranker = CrossEncoder(
                    self.reranker_model_name,
                    device="cpu",
                    max_length=512,
                )
            logger.info("Reranker model loaded on CPU (float32).")
        return self._reranker

    # ------------------------------------------------------------------ #
    # Index persistence
    # ------------------------------------------------------------------ #

    def save_index(self, directory: str | Path) -> None:
        """Persist FAISS index and BM25 state to disk."""
        import faiss

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(path / "faiss.index"))
        joblib.dump(
            {
                "bm25": self._bm25,
                "tokenised_corpus": self._tokenised_corpus,
                "chunks": self._chunks,
            },
            path / "bm25_state.joblib",
            compress=3,
        )
        logger.info("Index saved to %s", path)

    def load_index(self, directory: str | Path) -> None:
        """Load a previously saved index from disk."""
        import faiss

        path = Path(directory)
        self._faiss_index = faiss.read_index(str(path / "faiss.index"))
        state = joblib.load(path / "bm25_state.joblib")
        self._bm25 = state["bm25"]
        self._tokenised_corpus = state["tokenised_corpus"]
        self._chunks = state["chunks"]
        logger.info("Index loaded from %s (%d chunks).", path, len(self._chunks))

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """
        Simple whitespace + punctuation tokeniser for BM25.
        Lowercases and strips punctuation-only tokens.
        """
        import re

        tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
        return [t for t in tokens if len(t) > 1]

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
