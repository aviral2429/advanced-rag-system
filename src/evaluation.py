"""
evaluation.py — Quantitative evaluation harness for the RAG pipeline.

Improvement over baseline (no evaluation whatsoever):
- Recall@K: standard information-retrieval metric measuring how many of the
  truly relevant documents appear in the top-K retrieved results.
- MRR (Mean Reciprocal Rank): measures the quality of result *ordering*,
  not just presence.  A system that always puts the right answer first scores
  MRR=1.0; one that puts it tenth scores MRR=0.1.
- Faithfulness Score: cosine similarity between the generated answer embedding
  and the concatenated retrieved-context embedding.  Scores below 0.4 indicate
  the answer may contain material not grounded in the provided context
  (potential hallucination).
- Response latency: measured per query; essential for production systems and
  academic comparisons.
- JSONL experiment log: each evaluation run is appended to a log file so
  results are reproducible across sessions.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Computes retrieval and generation quality metrics for a RAG system.

    Parameters
    ----------
    embedding_model:
        Initialised EmbeddingModel (used for faithfulness scoring).
    log_path:
        Path to the JSONL file where metrics are appended.
    faithfulness_warning_threshold:
        Cosine similarity below which a hallucination warning is raised.
    """

    def __init__(
        self,
        embedding_model=None,
        log_path: str | Path = "eval_log.jsonl",
        faithfulness_warning_threshold: float = 0.4,
    ) -> None:
        self.embedding_model = embedding_model
        self.log_path = Path(log_path)
        self.faithfulness_warning_threshold = faithfulness_warning_threshold

    # ------------------------------------------------------------------ #
    # Retrieval metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str] | List[str],
        k: int,
    ) -> float:
        """
        Recall@K = |relevant ∩ top-K retrieved| / |relevant|

        Parameters
        ----------
        retrieved_ids:
            Ordered list of retrieved document identifiers (highest rank first).
        relevant_ids:
            Set of ground-truth relevant document identifiers.
        k:
            Cut-off rank.

        Returns
        -------
        float in [0, 1].  Returns 0.0 if relevant_ids is empty.
        """
        relevant_set = set(relevant_ids)
        if not relevant_set:
            return 0.0
        top_k = set(retrieved_ids[:k])
        return len(top_k & relevant_set) / len(relevant_set)

    @staticmethod
    def mean_reciprocal_rank(rankings: List[int]) -> float:
        """
        MRR = (1/N) * Σ_i  1/rank_i

        Parameters
        ----------
        rankings:
            List of 1-based ranks of the first relevant result for each query.
            A value of 0 means no relevant result was found.

        Returns
        -------
        float in [0, 1].
        """
        if not rankings:
            return 0.0
        reciprocals = [1.0 / r if r > 0 else 0.0 for r in rankings]
        return mean(reciprocals)

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str] | List[str],
        k: int,
    ) -> float:
        """
        Precision@K = |relevant ∩ top-K retrieved| / K
        """
        relevant_set = set(relevant_ids)
        if k == 0:
            return 0.0
        top_k = retrieved_ids[:k]
        return len(set(top_k) & relevant_set) / k

    # ------------------------------------------------------------------ #
    # Faithfulness / hallucination detection
    # ------------------------------------------------------------------ #

    def faithfulness_score(self, context: str, answer: str) -> float:
        """
        Estimate faithfulness as the cosine similarity between the answer
        embedding and the context embedding.

        Both vectors are L2-normalised (by EmbeddingModel), so cosine
        similarity equals the dot product.

        Interpretation:
            > 0.7  : high faithfulness — answer well grounded in context.
            0.4–0.7: borderline — some risk of hallucination.
            < 0.4  : low faithfulness — likely hallucination or off-topic.

        Returns
        -------
        float in [−1, 1].  For well-formed text and contexts this is
        typically in [0, 1].
        """
        if self.embedding_model is None:
            logger.warning("No embedding model set; returning default faithfulness of 1.0.")
            return 1.0

        try:
            ctx_vec = self.embedding_model.embed_query(context[:2000])  # truncate if huge
            ans_vec = self.embedding_model.embed_query(answer)
            return float(np.dot(ctx_vec, ans_vec))
        except Exception as exc:
            logger.error("Faithfulness score computation failed: %s", exc)
            return 1.0  # fail-open: don't falsely flag as hallucination

    def is_hallucination_risk(self, faithfulness: float) -> bool:
        return faithfulness < self.faithfulness_warning_threshold

    # ------------------------------------------------------------------ #
    # Latency measurement decorator
    # ------------------------------------------------------------------ #

    @staticmethod
    def measure_latency(func: Callable) -> Callable:
        """
        Decorator that wraps a function and returns (result, elapsed_ms).

        Usage::

            @RAGEvaluator.measure_latency
            def my_func():
                ...

            result, latency_ms = my_func()
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return result, elapsed_ms

        return wrapper

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def log_metrics(self, metrics: dict, extra: Optional[dict] = None) -> None:
        """
        Append a metrics dictionary to the JSONL evaluation log.

        Each line in the log is a valid JSON object containing the metrics
        and a UTC timestamp, enabling longitudinal experiment tracking.
        """
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metrics,
        }
        if extra:
            record.update(extra)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("Metrics logged to %s", self.log_path)

    # ------------------------------------------------------------------ #
    # Benchmark runner
    # ------------------------------------------------------------------ #

    def run_benchmark(
        self,
        pipeline,
        test_set: List[Dict],
        k_values: List[int] | None = None,
    ) -> Dict[str, float]:
        """
        Run a complete evaluation over a test set.

        Parameters
        ----------
        pipeline:
            Initialised RAGPipeline with an already-built index.
        test_set:
            List of dicts with keys:
            - 'question' (str)
            - 'gold_chunk_ids' (List[str]) — chunk doc_ids of relevant passages
            - 'reference_answer' (str, optional) — for faithfulness comparison
        k_values:
            List of K values for Recall@K.  Defaults to [5, 10].

        Returns
        -------
        dict with aggregated metrics.
        """
        if k_values is None:
            k_values = [5, 10]

        recall_scores: Dict[int, List[float]] = {k: [] for k in k_values}
        mrr_ranks: List[int] = []
        faithfulness_scores: List[float] = []
        latencies: List[float] = []

        for item in test_set:
            question = item["question"]
            gold_ids = set(item.get("gold_chunk_ids", []))

            start = time.perf_counter()
            result = pipeline.query(question)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            retrieved_ids = [sc.document.doc_id for sc in result.retrieved_chunks]

            # Recall@K
            for k in k_values:
                recall_scores[k].append(self.recall_at_k(retrieved_ids, gold_ids, k))

            # MRR — rank of first relevant result
            first_rank = 0
            for rank, doc_id in enumerate(retrieved_ids, start=1):
                if doc_id in gold_ids:
                    first_rank = rank
                    break
            mrr_ranks.append(first_rank)

            # Faithfulness
            context_text = " ".join(sc.document.content for sc in result.retrieved_chunks)
            faithfulness_scores.append(
                self.faithfulness_score(context_text, result.answer)
            )

        # Aggregate
        aggregated: Dict[str, float] = {}
        for k in k_values:
            aggregated[f"recall@{k}"] = mean(recall_scores[k]) if recall_scores[k] else 0.0
        aggregated["mrr"] = self.mean_reciprocal_rank(mrr_ranks)
        aggregated["mean_faithfulness"] = mean(faithfulness_scores) if faithfulness_scores else 0.0
        aggregated["mean_latency_ms"] = mean(latencies) if latencies else 0.0
        aggregated["num_queries"] = len(test_set)

        logger.info("Benchmark results: %s", aggregated)
        self.log_metrics(aggregated, extra={"event": "benchmark"})

        return aggregated
