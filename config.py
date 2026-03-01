"""
config.py — Centralized configuration for the Advanced RAG System.

All parameters are loaded from environment variables / .env file.
Pydantic BaseSettings provides automatic type validation and defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGConfig(BaseSettings):
    """
    Single source of truth for every tunable parameter in the RAG pipeline.
    Values are read from the .env file in the project root (or environment).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM Provider
    # ------------------------------------------------------------------
    llm_provider: Literal["together", "openai", "anthropic", "groq"] = Field(
        default="together",
        description="Which LLM API provider to use.",
    )

    # API Keys
    together_api_key: str = Field(default="", description="Together.ai API key.")
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    anthropic_api_key: str = Field(default="", description="Anthropic API key.")
    groq_api_key: str = Field(default="", description="Groq API key.")

    # Model names per provider
    together_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="Together.ai model identifier.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model identifier.",
    )
    anthropic_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Anthropic model identifier.",
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier.",
    )

    # Generation parameters
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sampling temperature; low = more deterministic.",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description="Maximum tokens in the generated answer.",
    )

    # ------------------------------------------------------------------
    # Embedding Model
    # ------------------------------------------------------------------
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="HuggingFace model name for dense embeddings.",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for encoding documents (CPU-safe default).",
    )

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        description="HuggingFace cross-encoder model for reranking.",
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to apply cross-encoder reranking after fusion.",
    )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = Field(
        default=1024,
        ge=128,
        le=4096,
        description="Target character size for each chunk.",
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=512,
        description="Character overlap between consecutive chunks.",
    )
    chunking_strategy: Literal["semantic", "recursive"] = Field(
        default="semantic",
        description="Chunking strategy: 'semantic' (preferred) or 'recursive' (baseline).",
    )
    semantic_breakpoint_percentile: int = Field(
        default=85,
        ge=50,
        le=99,
        description="Percentile threshold for semantic breakpoint detection.",
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to return after final reranking.",
    )
    retrieval_candidates: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Candidates retrieved from each of dense and sparse before fusion.",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="RRF constant k; controls rank smoothing.",
    )
    confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum normalized score; chunks below this are dropped.",
    )

    # ------------------------------------------------------------------
    # Hallucination / Faithfulness
    # ------------------------------------------------------------------
    faithfulness_warning_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Faithfulness score below which a hallucination warning is raised.",
    )

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    embed_cache_dir: Path = Field(
        default=Path(".cache/embeddings"),
        description="Directory for embedding cache files.",
    )
    index_cache_dir: Path = Field(
        default=Path(".cache/index"),
        description="Directory for persisted FAISS index and BM25 state.",
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    eval_log_path: Path = Field(
        default=Path("eval_log.jsonl"),
        description="Path to append JSONL evaluation logs.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("embed_cache_dir", "index_cache_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        return Path(v)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    def active_model(self) -> str:
        """Return the model name for the currently configured provider."""
        mapping = {
            "together": self.together_model,
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
            "groq": self.groq_model,
        }
        return mapping[self.llm_provider]

    def active_api_key(self) -> str:
        """Return the API key for the currently configured provider."""
        mapping = {
            "together": self.together_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
        }
        return mapping[self.llm_provider]
