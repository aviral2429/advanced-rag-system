"""
pipeline.py — End-to-end RAG orchestration with hallucination mitigation.

The RAGPipeline coordinates all modules (loader → chunker → embeddings →
retriever → LLM → evaluator) and returns a structured QueryResult containing
the answer, sources, confidence, faithfulness, and latency.

Hallucination mitigation strategy
----------------------------------
After generating the answer, the pipeline computes a faithfulness score
(cosine similarity between the answer embedding and the context embedding).
If the score falls below a configurable threshold, the `hallucination_warning`
flag in QueryResult is set to True and the Streamlit UI surfaces a visible
warning to the user.  This is a lightweight, reference-free method that does
not require a second API call for every query.

Improvement over baseline:
- All baseline improvements from every module are composed here.
- Lazy module initialisation: heavy models are loaded only on first use,
  so importing the pipeline does not trigger model downloads.
- Index persistence: call save() after indexing; load() on subsequent runs
  to skip re-embedding entirely.
- LRU caching of query results: repeated identical queries are answered
  instantly from cache without an API call.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import RAGConfig
from src.chunker import SemanticChunker
from src.embeddings import EmbeddingModel
from src.evaluation import RAGEvaluator
from src.hybrid_retriever import HybridRetriever, ScoredChunk
from src.llm import LLMClient
from src.loader import Document, PDFLoader

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured output from a single RAG query."""

    question: str
    answer: str
    sources: List[Dict]           # [{filename, page_number, score}]
    confidence: float             # mean retrieval score of top-K chunks
    faithfulness: float           # cosine sim(answer, context); 0–1
    hallucination_warning: bool   # True if faithfulness < threshold
    retrieved_chunks: List[ScoredChunk]
    latency_ms: float

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "faithfulness": self.faithfulness,
            "hallucination_warning": self.hallucination_warning,
            "latency_ms": self.latency_ms,
        }


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline: indexing and query serving.

    Parameters
    ----------
    config:
        RAGConfig instance with all runtime parameters.
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or RAGConfig()

        # Modules — initialised lazily on first use
        self._embedding_model: Optional[EmbeddingModel] = None
        self._retriever: Optional[HybridRetriever] = None
        self._chunker: Optional[SemanticChunker] = None
        self._llm: Optional[LLMClient] = None
        self._evaluator: Optional[RAGEvaluator] = None
        self._loader = PDFLoader()

        # State
        self._indexed_files: List[str] = []
        self._total_chunks: int = 0

    # ------------------------------------------------------------------ #
    # Lazy module accessors
    # ------------------------------------------------------------------ #

    @property
    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel(
                model_name=self.config.embedding_model,
                cache_dir=self.config.embed_cache_dir,
                batch_size=self.config.embedding_batch_size,
            )
        return self._embedding_model

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                embedding_model=self.embedding_model,
                rrf_k=self.config.rrf_k,
                use_reranker=self.config.use_reranker,
                reranker_model_name=self.config.reranker_model,
                candidates=self.config.retrieval_candidates,
                confidence_threshold=self.config.confidence_threshold,
            )
        return self._retriever

    @property
    def chunker(self) -> SemanticChunker:
        if self._chunker is None:
            self._chunker = SemanticChunker(
                embedding_model=self.embedding_model,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                strategy=self.config.chunking_strategy,
                breakpoint_percentile=self.config.semantic_breakpoint_percentile,
            )
        return self._chunker

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(
                provider=self.config.llm_provider,
                api_key=self.config.active_api_key(),
                model=self.config.active_model(),
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
        return self._llm

    @property
    def evaluator(self) -> RAGEvaluator:
        if self._evaluator is None:
            self._evaluator = RAGEvaluator(
                embedding_model=self.embedding_model,
                log_path=self.config.eval_log_path,
                faithfulness_warning_threshold=self.config.faithfulness_warning_threshold,
            )
        return self._evaluator

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def index(self, pdf_dir: str | Path) -> Dict:
        """
        Load, chunk, embed, and index all PDFs in *pdf_dir*.

        Returns a summary dict with indexing statistics.
        """
        start = time.perf_counter()

        logger.info("=== Indexing phase started ===")

        # Load documents
        documents: List[Document] = self._loader.load_directory(pdf_dir)
        if not documents:
            raise ValueError(f"No PDF pages loaded from {pdf_dir}.")

        unique_files = list({d.metadata["filename"] for d in documents})
        self._indexed_files = unique_files

        # Chunk
        chunks = self.chunker.chunk_documents(documents)
        self._total_chunks = len(chunks)

        # Build hybrid index
        self.retriever.build_index(chunks)

        elapsed = (time.perf_counter() - start) * 1000

        summary = {
            "num_files": len(unique_files),
            "num_pages": len(documents),
            "num_chunks": len(chunks),
            "indexing_time_ms": round(elapsed, 1),
        }
        logger.info("Indexing complete: %s", summary)
        return summary

    # ------------------------------------------------------------------ #
    # Query serving
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """
        Answer *question* using the indexed documents.

        Returns a QueryResult with answer, citations, confidence, faithfulness,
        latency, and an optional hallucination warning.
        """
        if top_k is None:
            top_k = self.config.top_k

        # Use cached result if question was already asked
        return self._query_cached(question, top_k)

    @functools.lru_cache(maxsize=128)
    def _query_cached(self, question: str, top_k: int) -> QueryResult:
        """Internal cached version of query().  LRU cache keyed on (question, top_k)."""
        return self._query_uncached(question, top_k)

    def _query_uncached(self, question: str, top_k: int) -> QueryResult:
        start = time.perf_counter()

        # 1. Retrieve
        retrieved_chunks: List[ScoredChunk] = self.retriever.retrieve(question, top_k=top_k)

        if not retrieved_chunks:
            logger.warning("No chunks retrieved for question: %r", question)
            return QueryResult(
                question=question,
                answer="The provided documents do not contain information about this topic.",
                sources=[],
                confidence=0.0,
                faithfulness=1.0,
                hallucination_warning=False,
                retrieved_chunks=[],
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # 2. Generate answer
        answer = self.llm.generate(question, retrieved_chunks)

        latency_ms = (time.perf_counter() - start) * 1000

        # 3. Compute faithfulness
        context_text = " ".join(sc.document.content for sc in retrieved_chunks)
        faithfulness = self.evaluator.faithfulness_score(context_text, answer)
        hallucination_warning = self.evaluator.is_hallucination_risk(faithfulness)

        if hallucination_warning:
            logger.warning(
                "Low faithfulness score (%.3f) detected for question: %r",
                faithfulness,
                question,
            )

        # 4. Build source list
        sources = self._build_sources(retrieved_chunks)

        # 5. Confidence = mean retrieval score
        confidence = sum(sc.score for sc in retrieved_chunks) / len(retrieved_chunks)

        # 6. Log metrics
        self.evaluator.log_metrics(
            {
                "question": question,
                "num_chunks_retrieved": len(retrieved_chunks),
                "confidence": round(confidence, 4),
                "faithfulness": round(faithfulness, 4),
                "hallucination_warning": hallucination_warning,
                "latency_ms": round(latency_ms, 1),
            },
            extra={"event": "query"},
        )

        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            confidence=round(confidence, 4),
            faithfulness=round(faithfulness, 4),
            hallucination_warning=hallucination_warning,
            retrieved_chunks=retrieved_chunks,
            latency_ms=round(latency_ms, 1),
        )

    # ------------------------------------------------------------------ #
    # Streaming variant
    # ------------------------------------------------------------------ #

    def stream_query(self, question: str, top_k: Optional[int] = None):
        """
        Generator variant of query() that yields answer tokens incrementally.

        Intended for use with Streamlit's st.write_stream().

        Yields
        ------
        str tokens from the LLM streaming API.

        Returns (via StopIteration value) a QueryResult with faithfulness
        and metadata — captured after the stream finishes.
        """
        if top_k is None:
            top_k = self.config.top_k

        start = time.perf_counter()

        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        if not retrieved_chunks:
            yield "The provided documents do not contain information about this topic."
            return

        full_answer = ""
        for token in self.llm.stream(question, retrieved_chunks):
            full_answer += token
            yield token

        latency_ms = (time.perf_counter() - start) * 1000
        context_text = " ".join(sc.document.content for sc in retrieved_chunks)
        faithfulness = self.evaluator.faithfulness_score(context_text, full_answer)

        self.evaluator.log_metrics(
            {
                "question": question,
                "faithfulness": round(faithfulness, 4),
                "latency_ms": round(latency_ms, 1),
            },
            extra={"event": "stream_query"},
        )

    # ------------------------------------------------------------------ #
    # Index persistence
    # ------------------------------------------------------------------ #

    def save(self, directory: str | Path | None = None) -> None:
        """Persist the FAISS + BM25 index to disk for fast reloading."""
        path = Path(directory) if directory else self.config.index_cache_dir
        self.retriever.save_index(path)
        logger.info("Pipeline index saved to %s", path)

    def load(self, directory: str | Path | None = None) -> None:
        """Load a previously persisted index, bypassing re-embedding."""
        path = Path(directory) if directory else self.config.index_cache_dir
        self.retriever.load_index(path)
        logger.info("Pipeline index loaded from %s", path)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_sources(chunks: List[ScoredChunk]) -> List[Dict]:
        seen = set()
        sources = []
        for sc in chunks:
            meta = sc.document.metadata
            key = (meta.get("filename"), meta.get("page_number"))
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "filename": meta.get("filename", "unknown"),
                        "page_number": meta.get("page_number", "?"),
                        "score": sc.score,
                        "retrieval_method": sc.retrieval_method,
                    }
                )
        return sources

    @property
    def is_indexed(self) -> bool:
        return self._retriever is not None and self._retriever.num_chunks > 0

    @property
    def stats(self) -> Dict:
        return {
            "indexed_files": self._indexed_files,
            "total_chunks": self._total_chunks,
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.active_model(),
            "embedding_model": self.config.embedding_model,
            "chunking_strategy": self.config.chunking_strategy,
            "use_reranker": self.config.use_reranker,
        }
