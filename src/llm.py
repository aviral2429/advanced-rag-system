"""
llm.py — Multi-provider LLM client with strict grounding and citation enforcement.

Improvement over baseline (unguided LLaMA2 generation):
- Strict system prompt prevents the model from hallucinating by explicitly
  prohibiting external knowledge and requiring citation of sources.
- Unified interface across Together.ai, OpenAI, and Anthropic: swap provider
  by changing a single environment variable.
- Streaming support: tokens are yielded as they are generated so the UI can
  display a live response rather than waiting for full completion.
- Tenacity retry with exponential back-off handles transient API errors
  transparently (rate limits, 5xx errors).
- Context-window guard: estimates token count and trims lowest-ranked chunks
  before sending to avoid silent truncation by the API.
"""

from __future__ import annotations

import logging
from typing import Iterator, List

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.hybrid_retriever import ScoredChunk

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# System prompt — strict grounding rules
# -----------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a research assistant whose only source of knowledge \
is the Context provided below.

RULES (follow ALL of them without exception):
1. Answer ONLY using information explicitly stated in the Context.
2. If the answer is not present in the Context, respond EXACTLY:
   "The provided documents do not contain information about this topic."
3. Cite every factual claim using the format:
   [Source: <filename>, Page: <page_number>]
4. Do NOT speculate, infer beyond what is written, or use external knowledge.
5. Be concise, precise, and factual.

Context:
{context}
"""

_MAX_CONTEXT_TOKENS = 6000  # conservative limit shared across providers


class LLMClient:
    """
    Provider-agnostic LLM client that enforces grounded generation.

    Parameters
    ----------
    provider:
        One of 'together', 'openai', 'anthropic'.
    api_key:
        API key for the chosen provider.
    model:
        Model identifier (provider-specific).
    temperature:
        Sampling temperature (0.0–1.0).  Low values = deterministic.
    max_tokens:
        Maximum tokens to generate.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, question: str, context_chunks: List[ScoredChunk]) -> str:
        """
        Generate a grounded answer to *question* using *context_chunks*.

        Returns the full response string.
        """
        messages = self._build_messages(question, context_chunks)
        return self._call(messages)

    def stream(
        self, question: str, context_chunks: List[ScoredChunk]
    ) -> Iterator[str]:
        """
        Stream the answer token-by-token.

        Yields incremental text fragments (suitable for Streamlit's
        ``st.write_stream``).
        """
        messages = self._build_messages(question, context_chunks)
        yield from self._call_streaming(messages)

    # ------------------------------------------------------------------ #
    # Message construction
    # ------------------------------------------------------------------ #

    def _build_messages(
        self, question: str, context_chunks: List[ScoredChunk]
    ) -> List[dict]:
        """
        Construct the API message list with the strict system prompt and
        formatted context, trimmed to fit within the context window.
        """
        # Trim chunks if necessary
        trimmed_chunks = self._trim_context(context_chunks)
        formatted_context = self._format_context(trimmed_chunks)

        system_content = _SYSTEM_PROMPT.format(context=formatted_context)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

    @staticmethod
    def _format_context(chunks: List[ScoredChunk]) -> str:
        """
        Format retrieved chunks into a clearly labelled context block.

        Each chunk is labelled with its source filename, page number, and
        retrieval rank so the LLM can emit accurate citations.
        """
        parts: List[str] = []
        for i, sc in enumerate(chunks, start=1):
            meta = sc.document.metadata
            filename = meta.get("filename", "unknown")
            page = meta.get("page_number", "?")
            score = sc.score

            header = f"[Passage {i} | Source: {filename} | Page: {page} | Score: {score:.3f}]"
            parts.append(f"{header}\n{sc.document.content.strip()}")

        return "\n\n---\n\n".join(parts) if parts else "(no context retrieved)"

    def _trim_context(self, chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        """
        Remove lowest-ranked chunks until the estimated token count fits
        within _MAX_CONTEXT_TOKENS.  Preserves top-ranked (most relevant)
        chunks.
        """
        trimmed: List[ScoredChunk] = []
        total_tokens = 0

        for sc in chunks:  # already sorted highest→lowest by pipeline
            est = self._estimate_tokens(sc.document.content)
            if total_tokens + est > _MAX_CONTEXT_TOKENS:
                logger.debug(
                    "Context trimmed: dropped chunk from %s (budget exceeded).",
                    sc.document.metadata.get("filename", "?"),
                )
                break
            trimmed.append(sc)
            total_tokens += est

        return trimmed

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: 4 characters ≈ 1 token (GPT-4 rule of thumb)."""
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------ #
    # Provider dispatch — non-streaming
    # ------------------------------------------------------------------ #

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call(self, messages: List[dict]) -> str:
        if self.provider == "together":
            return self._call_together(messages)
        elif self.provider == "openai":
            return self._call_openai(messages)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages)
        elif self.provider == "groq":
            return self._call_groq(messages)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    def _call_together(self, messages: List[dict]) -> str:
        """Together.ai uses an OpenAI-compatible API endpoint."""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.together.xyz/v1",
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_openai(self, messages: List[dict]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_groq(self, messages: List[dict]) -> str:
        """Groq uses an OpenAI-compatible API endpoint."""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, messages: List[dict]) -> str:
        import anthropic

        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user_messages = [m for m in messages if m["role"] != "system"]

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            system=system_msg,
            messages=user_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.content[0].text

    # ------------------------------------------------------------------ #
    # Provider dispatch — streaming
    # ------------------------------------------------------------------ #

    def _call_streaming(self, messages: List[dict]) -> Iterator[str]:
        if self.provider == "together":
            yield from self._stream_openai_compatible(
                messages,
                base_url="https://api.together.xyz/v1",
            )
        elif self.provider == "openai":
            yield from self._stream_openai_compatible(messages, base_url=None)
        elif self.provider == "anthropic":
            yield from self._stream_anthropic(messages)
        elif self.provider == "groq":
            yield from self._stream_openai_compatible(
                messages,
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    def _stream_openai_compatible(
        self, messages: List[dict], base_url: str | None
    ) -> Iterator[str]:
        from openai import OpenAI

        kwargs = {"api_key": self.api_key}
        if base_url:
            kwargs["base_url"] = base_url

        client = OpenAI(**kwargs)
        with client.chat.completions.stream(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _stream_anthropic(self, messages: List[dict]) -> Iterator[str]:
        import anthropic

        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user_messages = [m for m in messages if m["role"] != "system"]

        client = anthropic.Anthropic(api_key=self.api_key)
        with client.messages.stream(
            model=self.model,
            system=system_msg,
            messages=user_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text
