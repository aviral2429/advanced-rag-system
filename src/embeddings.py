"""
embeddings.py — Dense embedding model wrapper using BAAI/bge-large-en-v1.5.

Improvement over baseline:
- BAAI/bge-large-en-v1.5 ranks in the top tier on the MTEB benchmark,
  significantly outperforming all-MiniLM (the typical baseline).
- BGE-specific asymmetric query instruction prefix boosts retrieval accuracy.
- L2 normalization of all vectors enables cosine similarity via a dot product
  (FAISS IndexFlatIP), which is faster and exact on CPU.
- Disk-based joblib cache: re-indexing the same PDFs skips re-embedding,
  saving minutes on repeated runs.
- CPU-safe batch size (default 32) prevents out-of-memory errors on machines
  without a GPU.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# BGE models require a task-specific prefix on *queries* (not documents).
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Models that use the BGE query prefix
_BGE_MODELS = {"BAAI/bge-large-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5"}


class EmbeddingModel:
    """
    Wraps a sentence-transformers model for encoding both documents and queries.

    The model is loaded lazily on first use so that importing this module
    (e.g., for type hints) does not trigger a large model download.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. 'BAAI/bge-large-en-v1.5').
    cache_dir:
        Directory for storing serialised embedding arrays (joblib format).
    batch_size:
        Sentences per encoding batch.  32 is safe for CPU; raise to 64–128
        on a GPU.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        cache_dir: str | Path = ".cache/embeddings",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        # Detect whether to use the BGE query prefix
        self._use_bge_prefix = model_name in _BGE_MODELS

        # Model loaded lazily
        self._model: Optional[object] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def embed_documents(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of document texts into L2-normalised dense vectors.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim), dtype float32.
        """
        if not texts:
            return np.empty((0, self._dim()), dtype=np.float32)

        cache_key = self._get_cache_key(texts)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            logger.info("Embedding cache hit (%d vectors).", len(texts))
            return cached

        logger.info("Encoding %d document(s) with %s …", len(texts), self.model_name)
        vectors = self._encode_batched(texts, prefix=None, show_progress=show_progress)
        vectors = self._normalize(vectors)

        self._save_to_cache(cache_key, vectors)
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        """
        Encode a single query string into an L2-normalised dense vector.

        BGE models benefit from the task-specific instruction prefix on
        queries (but NOT on documents).

        Returns
        -------
        np.ndarray of shape (embedding_dim,), dtype float32.
        """
        prefix = _BGE_QUERY_PREFIX if self._use_bge_prefix else None
        query_text = f"{prefix}{text}" if prefix else text
        vectors = self._encode_batched([query_text], prefix=None, show_progress=False)
        return self._normalize(vectors)[0]

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality (e.g. 1024 for bge-large-en-v1.5)."""
        return self._dim()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_model(self):
        """Lazy-load the sentence-transformers model.

        Strategy:
        1. Try to load from local HuggingFace cache (no network needed).
        2. Fall back to full download with retries.
        3. Raise a clear RuntimeError if both fail.

        CPU-safety: device_map=None and low_cpu_mem_usage=False are passed via
        model_kwargs to prevent sentence-transformers 3.x + accelerate from
        loading model layers onto the PyTorch 'meta' device, which causes the
        'Cannot copy out of meta tensor; no data!' error on CPU-only machines.
        """
        if self._model is None:
            import os
            from sentence_transformers import SentenceTransformer  # lazy import

            # Prevent accelerate / tokenizers from interfering with CPU loading
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("ACCELERATE_USE_FSDP", "0")

            device = self._detect_device()
            logger.info("Loading embedding model '%s' on device '%s'.", self.model_name, device)

            # Kwargs that prevent 'meta tensor' issues on CPU-only systems.
            _safe_model_kwargs = {
                "device_map": None,
                "low_cpu_mem_usage": False,
            }

            # --- Pass 1: try local cache only (fast, works offline) ---
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=device,
                    local_files_only=True,
                    model_kwargs=_safe_model_kwargs,
                )
                logger.info("Loaded embedding model from local cache.")
                return self._model
            except TypeError:
                # Older sentence-transformers doesn't support model_kwargs
                try:
                    self._model = SentenceTransformer(
                        self.model_name, device=device, local_files_only=True
                    )
                    logger.info("Loaded embedding model from local cache (compat mode).")
                    return self._model
                except Exception:
                    pass
            except Exception:
                logger.info("Model not in local cache; attempting download …")

            # --- Pass 2: full download ---
            try:
                try:
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=device,
                        model_kwargs=_safe_model_kwargs,
                    )
                except TypeError:
                    # Fallback for older sentence-transformers versions
                    self._model = SentenceTransformer(self.model_name, device=device)
                logger.info("Embedding model downloaded and loaded successfully.")
            except Exception as exc:
                raise RuntimeError(
                    f"\n\n❌ Could not load embedding model: '{self.model_name}'\n"
                    f"   Reason: {exc}\n\n"
                    "   Solutions:\n"
                    "   1. Check your internet connection and try again.\n"
                    "   2. Pre-download the model manually:\n"
                    f"      python -c \"from sentence_transformers import SentenceTransformer; "
                    f"SentenceTransformer('{self.model_name}')\"\n"
                    "   3. Switch to a smaller model in .env:\n"
                    "      EMBEDDING_MODEL=all-MiniLM-L6-v2\n"
                ) from exc

        return self._model

    def _detect_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _encode_batched(
        self,
        texts: List[str],
        prefix: Optional[str],
        show_progress: bool,
    ) -> np.ndarray:
        """Encode texts in batches and concatenate results."""
        model = self._get_model()
        all_vectors: List[np.ndarray] = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", unit="batch")

        for start in iterator:
            batch = texts[start : start + self.batch_size]
            if prefix:
                batch = [f"{prefix}{t}" for t in batch]
            vecs = model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,  # we normalise ourselves
            )
            all_vectors.append(vecs.astype(np.float32))

        return np.vstack(all_vectors)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise rows so cosine similarity equals dot product."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        return (vectors / norms).astype(np.float32)

    def _get_cache_key(self, texts: List[str]) -> str:
        """
        Deterministic cache key: MD5 of model_name + sorted text content.
        Sorting makes the key independent of retrieval order.
        """
        payload = self.model_name + "".join(sorted(texts))
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.joblib"

    def _load_from_cache(self, key: str) -> Optional[np.ndarray]:
        path = self._cache_path(key)
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as exc:
                logger.warning("Cache load failed (%s); re-computing.", exc)
        return None

    def _save_to_cache(self, key: str, vectors: np.ndarray) -> None:
        path = self._cache_path(key)
        try:
            joblib.dump(vectors, path, compress=3)
            logger.debug("Embeddings cached to %s", path)
        except Exception as exc:
            logger.warning("Cache save failed: %s", exc)

    def _dim(self) -> int:
        """Return the embedding dimension by loading the model."""
        return self._get_model().get_sentence_embedding_dimension()
