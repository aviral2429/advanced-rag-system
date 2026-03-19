"""
evaluation.py — RAG Precision Matrix
Calculates 4 core RAG quality metrics for each Q&A pair:
  1. Context Precision   — Are retrieved chunks relevant to the question?
  2. Answer Faithfulness — Is the answer grounded in retrieved context?
  3. Answer Relevancy    — Does the answer actually address the question?
  4. Context Recall      — Does retrieved context cover the expected answer?
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Shared lightweight model (same family as embeddings.py, loaded once)
_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _encode(texts: List[str]) -> np.ndarray:
    return _get_model().encode(texts, normalize_embeddings=True)


# ── Individual metrics ────────────────────────────────────────────────────────

def context_precision(question: str, contexts: List[str], threshold: float = 0.35) -> float:
    """
    Fraction of retrieved chunks that are semantically relevant to the question.
    Score = (# relevant chunks) / (# total chunks retrieved)
    """
    if not contexts:
        return 0.0
    q_emb = _encode([question])[0]
    c_embs = _encode(contexts)
    scores = [_cosine(q_emb, c) for c in c_embs]
    relevant = sum(1 for s in scores if s >= threshold)
    return round(relevant / len(contexts), 4)


def answer_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    How well the answer is grounded in the retrieved context.
    Score = max cosine similarity between answer and any context chunk.
    """
    if not contexts or not answer.strip():
        return 0.0
    a_emb = _encode([answer])[0]
    c_embs = _encode(contexts)
    scores = [_cosine(a_emb, c) for c in c_embs]
    return round(max(scores), 4)


def answer_relevancy(question: str, answer: str) -> float:
    """
    How directly the answer addresses the question.
    Score = cosine similarity between question and answer embeddings.
    """
    if not answer.strip():
        return 0.0
    embs = _encode([question, answer])
    return round(_cosine(embs[0], embs[1]), 4)


def context_recall(expected_answer: str, contexts: List[str]) -> float:
    """
    How well the retrieved context covers the expected answer.
    Score = max cosine similarity between expected answer and any context chunk.
    """
    if not contexts or not expected_answer.strip():
        return 0.0
    ea_emb = _encode([expected_answer])[0]
    c_embs = _encode(contexts)
    scores = [_cosine(ea_emb, c) for c in c_embs]
    return round(max(scores), 4)


# ── Full evaluation ───────────────────────────────────────────────────────────

def evaluate_row(
    question: str,
    answer: str,
    expected_answer: str,
    source_docs: list,
) -> dict:
    """
    Evaluate a single Q&A pair and return a dict of all 4 metrics + overall score.
    source_docs: list of LangChain Document objects returned by the retriever.
    """
    contexts = [doc.page_content for doc in source_docs]

    cp = context_precision(question, contexts)
    af = answer_faithfulness(answer, contexts)
    ar = answer_relevancy(question, answer)
    cr = context_recall(expected_answer, contexts)
    overall = round((cp + af + ar + cr) / 4, 4)

    return {
        "Question": question,
        "Expected Answer": expected_answer,
        "Generated Answer": answer,
        "Context Precision": cp,
        "Answer Faithfulness": af,
        "Answer Relevancy": ar,
        "Context Recall": cr,
        "Overall Score": overall,
    }


def evaluate_batch(
    qa_pairs: List[Tuple[str, str]],   # list of (question, expected_answer)
    qa_bundle: dict,                    # from build_qa_chain()
) -> List[dict]:
    """
    Run evaluation on a list of (question, expected_answer) pairs.
    Returns a list of result dicts (one per pair).
    """
    from qa_system import ask_question

    results = []
    for question, expected_answer in qa_pairs:
        answer, source_docs = ask_question(qa_bundle, question)
        row = evaluate_row(question, answer, expected_answer, source_docs)
        results.append(row)
    return results
