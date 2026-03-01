"""
src — Advanced RAG System package.

Exposes the primary public surface:
    RAGPipeline  — end-to-end orchestrator
    RAGConfig    — configuration object
"""

from config import RAGConfig
from src.pipeline import QueryResult, RAGPipeline

__all__ = ["RAGConfig", "RAGPipeline", "QueryResult"]
