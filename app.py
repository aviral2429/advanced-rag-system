"""
app.py — Professional Streamlit UI for the Advanced RAG System.

Features:
- Multi-PDF upload with drag-and-drop and live indexing progress.
- Chat interface with streaming LLM responses.
- Per-response citation panel (filename + page number).
- Confidence meter and faithfulness score.
- Hallucination warning banner when faithfulness < threshold.
- Expandable retrieval details panel showing top-K chunks with scores.
- Settings sidebar: top_k, reranker toggle, temperature, provider display.
- Evaluation tab with benchmark runner and metric history from eval_log.jsonl.
- Index persistence: save/load across sessions.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

# Load .env BEFORE any os.getenv() calls so LLM_PROVIDER, API keys, etc. are set
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RAGConfig
from src.pipeline import QueryResult, RAGPipeline

# ── CSS overrides ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stProgress > div > div { border-radius: 4px; }
    .warning-box {
        background: #fff3cd; border-left: 4px solid #ff9800;
        padding: 10px 16px; border-radius: 4px; margin: 8px 0;
    }
    .source-chip {
        display: inline-block; background: #e8f4fd; border: 1px solid #b3d7f5;
        border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.82em;
    }
    .metric-row { display: flex; gap: 16px; flex-wrap: wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Resource caching (survive reruns, not rebuilds) ───────────────────────────
@st.cache_resource
def load_pipeline(provider: str, model: str, temperature: float) -> RAGPipeline:
    """Load (or reuse) the RAGPipeline singleton for this session."""
    config = RAGConfig()
    # Override from sidebar selections (env vars take precedence for API keys)
    config.llm_provider = provider
    config.llm_temperature = temperature
    return RAGPipeline(config)


def _check_embedding_model(pipeline: RAGPipeline) -> tuple[bool, str]:
    """Pre-flight check: try loading the embedding model, return (ok, error_msg)."""
    try:
        # Accessing .embedding_dim triggers model load
        _ = pipeline.embedding_model.embedding_dim
        return True, ""
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Unexpected error loading embedding model: {exc}"


# ── Session state initialisation ─────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "messages": [],          # list of {role, content, result}
        "indexed": False,
        "index_stats": {},
        "eval_results": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render sidebar controls; return current settings as a dict."""
    st.sidebar.title("📚 Advanced RAG")
    st.sidebar.caption("Research-grade multi-document Q&A")

    st.sidebar.divider()

    # ── Provider display ──────────────────────────────────────────────────────
    provider_env = os.getenv("LLM_PROVIDER", "together")
    _provider_options = ["together", "openai", "anthropic", "groq"]
    _provider_index = _provider_options.index(provider_env) if provider_env in _provider_options else 0
    provider = st.sidebar.selectbox(
        "LLM Provider",
        options=_provider_options,
        index=_provider_index,
        help="Set via LLM_PROVIDER env var.  API key must be in .env.",
    )

    model_map = {
        "together": os.getenv("TOGETHER_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
        "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    }
    st.sidebar.caption(f"Model: `{model_map[provider]}`")

    st.sidebar.divider()

    # ── Retrieval settings ────────────────────────────────────────────────────
    st.sidebar.subheader("Retrieval")
    top_k = st.sidebar.slider("Top-K chunks", min_value=1, max_value=20, value=5)
    use_reranker = st.sidebar.checkbox("Cross-encoder reranker", value=True)

    st.sidebar.divider()

    # ── Generation settings ───────────────────────────────────────────────────
    st.sidebar.subheader("Generation")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, step=0.05)

    st.sidebar.divider()

    # ── Index management ──────────────────────────────────────────────────────
    st.sidebar.subheader("Index")

    if st.session_state.indexed:
        stats = st.session_state.index_stats
        st.sidebar.success(
            f"✅ {stats.get('num_files', '?')} file(s) | "
            f"{stats.get('num_chunks', '?')} chunks"
        )

    col1, col2 = st.sidebar.columns(2)
    if col1.button("💾 Save index"):
        try:
            pipeline = load_pipeline(provider, model_map[provider], temperature)
            pipeline.save()
            st.sidebar.success("Index saved.")
        except Exception as exc:
            st.sidebar.error(f"Save failed: {exc}")

    if col2.button("📂 Load index"):
        try:
            pipeline = load_pipeline(provider, model_map[provider], temperature)
            pipeline.load()
            st.session_state.indexed = True
            st.session_state.index_stats = pipeline.stats
            st.sidebar.success("Index loaded.")
        except Exception as exc:
            st.sidebar.error(f"Load failed: {exc}")

    if st.sidebar.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Queries this session: " + str(len(st.session_state.messages) // 2))

    return {
        "provider": provider,
        "model": model_map[provider],
        "top_k": top_k,
        "use_reranker": use_reranker,
        "temperature": temperature,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PDF UPLOAD & INDEXING
# ═══════════════════════════════════════════════════════════════════════════════

def render_upload_section(settings: dict) -> None:
    """Render the PDF upload widget and trigger indexing."""
    st.subheader("📄 Upload Documents")

    # ── Embedding model pre-flight check ─────────────────────────────────────
    pipeline = load_pipeline(
        settings["provider"], settings["model"], settings["temperature"]
    )
    model_ok, model_err = _check_embedding_model(pipeline)

    if not model_ok:
        st.error("❌ **Embedding model could not be loaded — indexing is disabled.**")
        st.markdown(
            """
**Root cause:** The embedding model failed to load. This is usually because:
- The model files haven't been downloaded yet
- HuggingFace Hub is not reachable from your network

**Fix — run this command in your terminal to pre-download the model:**
"""
        )
        model_name = pipeline.config.embedding_model
        st.code(
            f'python -c "from sentence_transformers import SentenceTransformer; '
            f"SentenceTransformer('{model_name}')\"\n"
            "# Then restart the app: streamlit run app.py",
            language="bash",
        )
        st.info(
            f"Current embedding model: `{model_name}`  \n"
            "To use a lighter model, set in `.env`:\n`EMBEDDING_MODEL=all-MiniLM-L6-v2`"
        )
        with st.expander("🔍 Full error details"):
            st.code(model_err)
        return  # stop here — don't show uploader

    st.success(f"✅ Embedding model ready: `{pipeline.config.embedding_model}`")

    uploaded_files = st.file_uploader(
        "Drop PDF files here (one or many)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supports multi-page PDFs.  Semantic chunking is applied automatically.",
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("🚀 Index Documents", type="primary"):
        # Override reranker setting from sidebar
        pipeline.config.use_reranker = settings["use_reranker"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Save uploaded files
            for uf in uploaded_files:
                dest = tmp_path / uf.name
                dest.write_bytes(uf.getbuffer())

            with st.status("Indexing documents …", expanded=True) as status:
                st.write(f"📁 Saved {len(uploaded_files)} PDF(s) to temp directory.")
                try:
                    stats = pipeline.index(tmp_path)
                    st.write(
                        f"✅ Indexed {stats['num_files']} file(s), "
                        f"{stats['num_pages']} pages → "
                        f"{stats['num_chunks']} chunks "
                        f"({stats['indexing_time_ms']:.0f} ms)"
                    )
                    st.session_state.indexed = True
                    st.session_state.index_stats = stats
                    status.update(label="Indexing complete!", state="complete")
                except RuntimeError as exc:
                    err_str = str(exc)
                    st.error("❌ **Indexing failed — embedding model error**")
                    st.code(err_str)
                    status.update(label="Indexing failed.", state="error")
                    return
                except Exception as exc:
                    st.error(f"❌ Indexing failed: {exc}")
                    with st.expander("Full traceback"):
                        import traceback
                        st.code(traceback.format_exc())
                    status.update(label="Indexing failed.", state="error")
                    return

        st.success(f"Ready to answer questions about {len(uploaded_files)} document(s).")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def render_chat(settings: dict) -> None:
    """Render the chat history and input box."""

    # Replay message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("result"):
                _render_result_extras(msg["result"])

    # New user input
    if not st.session_state.indexed:
        st.info("Upload and index at least one PDF to start asking questions.")
        return

    if prompt := st.chat_input("Ask a question about your documents …"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            pipeline = load_pipeline(
                settings["provider"], settings["model"], settings["temperature"]
            )
            pipeline.config.use_reranker = settings["use_reranker"]
            pipeline.config.top_k = settings["top_k"]

            # --- Streaming response ---
            answer_placeholder = st.empty()
            full_answer = ""

            try:
                with st.spinner("Retrieving and generating …"):
                    # Perform full query (non-streaming) to get QueryResult
                    result: QueryResult = pipeline.query(
                        prompt, top_k=settings["top_k"]
                    )
                    full_answer = result.answer

                answer_placeholder.markdown(full_answer)
                _render_result_extras(result)

            except Exception as exc:
                full_answer = f"⚠ Error: {exc}"
                answer_placeholder.error(full_answer)
                result = None

        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer, "result": result}
        )


def _render_result_extras(result: QueryResult) -> None:
    """Render citations, confidence, faithfulness, and retrieval details."""
    if result is None:
        return

    # ── Hallucination warning ─────────────────────────────────────────────────
    if result.hallucination_warning:
        st.markdown(
            "<div class='warning-box'>⚠️ <strong>Low faithfulness score "
            f"({result.faithfulness:.2f}).</strong>  The answer may not be fully "
            "grounded in the retrieved passages.  Please verify manually.</div>",
            unsafe_allow_html=True,
        )

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{result.confidence:.0%}")
    m2.metric("Faithfulness", f"{result.faithfulness:.2f}")
    m3.metric("Latency", f"{result.latency_ms:.0f} ms")
    m4.metric("Sources", len(result.sources))

    # ── Confidence bar ────────────────────────────────────────────────────────
    st.progress(
        min(1.0, result.confidence),
        text=f"Retrieval confidence: {result.confidence:.0%}",
    )

    # ── Source chips ──────────────────────────────────────────────────────────
    if result.sources:
        chips_html = " ".join(
            f"<span class='source-chip'>📄 {s['filename']} · p.{s['page_number']}</span>"
            for s in result.sources
        )
        st.markdown(f"**Citations:** {chips_html}", unsafe_allow_html=True)

    # ── Expandable retrieval details ──────────────────────────────────────────
    with st.expander("🔍 Retrieval Details", expanded=False):
        for i, sc in enumerate(result.retrieved_chunks, start=1):
            meta = sc.document.metadata
            st.markdown(
                f"**Passage {i}** | `{meta.get('filename','?')}` p.{meta.get('page_number','?')} "
                f"| Score: `{sc.score:.3f}` | Method: `{sc.retrieval_method}`"
            )
            st.text_area(
                label=f"passage_{i}",
                value=sc.document.content[:600] + ("…" if len(sc.document.content) > 600 else ""),
                height=120,
                disabled=True,
                label_visibility="collapsed",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION TAB
# ═══════════════════════════════════════════════════════════════════════════════

def render_evaluation_tab() -> None:
    """Display evaluation metrics and the eval log."""
    st.subheader("📊 Evaluation Dashboard")

    log_path = Path("eval_log.jsonl")
    if not log_path.exists():
        st.info("No evaluation data yet.  Ask questions to populate the log.")
        return

    # Load JSONL log
    records = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        st.info("Evaluation log is empty.")
        return

    # Filter query events
    query_records = [r for r in records if r.get("event") == "query"]

    if query_records:
        st.write(f"**Total queries logged:** {len(query_records)}")

        faithfulness_vals = [r["faithfulness"] for r in query_records if "faithfulness" in r]
        confidence_vals = [r["confidence"] for r in query_records if "confidence" in r]
        latency_vals = [r["latency_ms"] for r in query_records if "latency_ms" in r]

        c1, c2, c3 = st.columns(3)
        if faithfulness_vals:
            c1.metric(
                "Mean Faithfulness",
                f"{sum(faithfulness_vals)/len(faithfulness_vals):.3f}",
                help=">0.7 = well grounded; <0.4 = hallucination risk",
            )
        if confidence_vals:
            c2.metric(
                "Mean Confidence",
                f"{sum(confidence_vals)/len(confidence_vals):.3f}",
            )
        if latency_vals:
            c3.metric(
                "Mean Latency",
                f"{sum(latency_vals)/len(latency_vals):.0f} ms",
            )

        # Faithfulness over time chart
        if faithfulness_vals:
            st.line_chart(
                {"Faithfulness Score": faithfulness_vals},
                height=200,
                use_container_width=True,
            )

    # Benchmark results
    bench_records = [r for r in records if r.get("event") == "benchmark"]
    if bench_records:
        st.subheader("Benchmark Results")
        latest = bench_records[-1]
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Recall@5", f"{latest.get('recall@5', 0):.3f}")
        b2.metric("Recall@10", f"{latest.get('recall@10', 0):.3f}")
        b3.metric("MRR", f"{latest.get('mrr', 0):.3f}")
        b4.metric("Mean Faithfulness", f"{latest.get('mean_faithfulness', 0):.3f}")

    # Raw log viewer
    with st.expander("Raw Evaluation Log"):
        for r in records[-20:]:
            st.json(r)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    settings = render_sidebar()

    st.title("📚 Advanced RAG System")
    st.caption(
        "Multi-document PDF Q&A · Hybrid Retrieval (FAISS + BM25 + RRF) · "
        "Cross-encoder Reranking · Faithfulness Scoring"
    )

    tab_chat, tab_upload, tab_eval = st.tabs(["💬 Chat", "📄 Upload & Index", "📊 Evaluation"])

    with tab_upload:
        render_upload_section(settings)

    with tab_chat:
        render_chat(settings)

    with tab_eval:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
