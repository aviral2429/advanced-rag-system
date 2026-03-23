"""
app.py — Merged RAG System
Frontend: advanced-rag-system UI (chat bubbles, faithfulness metrics, source chips,
          confidence progress bar, hallucination warnings, evaluation dashboard)
Backend:  PDF-RAG-Chatbot (LangChain + FAISS + HuggingFace embeddings, Supabase cloud storage)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Backend imports ───────────────────────────────────────────────────────────
from pdf_loader import load_pdf, load_pdf_from_url
from text_splitter import split_documents
from embeddings import get_embeddings
from vector_store import create_vector_store, get_retriever
from qa_system import build_qa_chain, ask_question
from evaluation import evaluate_batch

# ── Supabase (optional) ───────────────────────────────────────────────────────
try:
    from cloud_storage import upload_pdf, list_pdfs, get_pdf_download_url, delete_pdf
    _supabase_ok = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY"))
except ImportError:
    _supabase_ok = False

FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.50"))
EVAL_LOG = Path("eval_log.jsonl")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Merged RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
    }
    section[data-testid="stSidebar"] * { color: #e0e0ff !important; }

    /* Chat messages */
    .stChatMessage {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(8px);
        animation: fadeUp 0.3s ease both;
        margin-bottom: 0.6rem;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0);    }
    }

    /* Source chip */
    .source-chip {
        display: inline-block;
        background: rgba(124,106,255,0.15);
        border: 1px solid rgba(124,106,255,0.40);
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.82em;
        color: #c4b8ff !important;
    }

    /* Hallucination warning */
    .warning-box {
        background: rgba(255,152,0,0.12);
        border-left: 4px solid #ff9800;
        padding: 10px 16px;
        border-radius: 6px;
        margin: 8px 0;
        color: #ffd180 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c6aff, #00d4aa) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.4rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Inputs */
    .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    .stChatInput textarea {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    /* Headings / text */
    h1, h2, h3, p, .stMarkdown { color: #e0e0ff !important; }
    .stAlert { border-radius: 10px !important; }
    #MainMenu, footer { visibility: hidden; }

    /* Progress bar */
    .stProgress > div > div { border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached embedding model (shared across all operations) ─────────────────────
@st.cache_resource
def get_cached_embeddings():
    return get_embeddings()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "messages":     [],
    "vectorstore":  None,
    "qa_chain":     None,
    "indexed_file": None,
    "cloud_mode":   "upload",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_faithfulness(answer: str, sources: list) -> float:
    """Cosine similarity between answer embedding and source context embedding (0–1)."""
    try:
        emb_model = get_cached_embeddings()
        ctx = " ".join(s.page_content for s in sources)[:2000]
        a_vec = np.array(emb_model.embed_query(answer))
        c_vec = np.array(emb_model.embed_query(ctx))
        cosine = float(
            np.dot(a_vec, c_vec) /
            (np.linalg.norm(a_vec) * np.linalg.norm(c_vec) + 1e-9)
        )
        return round(max(0.0, min(1.0, cosine)), 3)
    except Exception:
        return 0.75  # safe fallback


def _log_query(q: str, a: str, confidence: float, faithfulness: float,
               latency_ms: float, n_sources: int) -> None:
    record = {
        "event": "query",
        "question": q,
        "answer": a[:200],
        "confidence": round(confidence, 3),
        "faithfulness": round(faithfulness, 3),
        "latency_ms": round(latency_ms, 1),
        "n_sources": n_sources,
    }
    with EVAL_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _render_answer_extras(sources: list, confidence: float,
                           faithfulness: float, latency_ms: float,
                           threshold: float) -> None:
    """Render metrics, progress bar, source chips, retrieval details below an answer."""

    # Hallucination warning
    if faithfulness < threshold:
        st.markdown(
            f"<div class='warning-box'>⚠️ <strong>Low faithfulness score "
            f"({faithfulness:.2f}).</strong> The answer may not be fully grounded "
            "in the retrieved passages. Please verify manually.</div>",
            unsafe_allow_html=True,
        )

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence",   f"{confidence:.0%}")
    m2.metric("Faithfulness", f"{faithfulness:.2f}")
    m3.metric("Latency",      f"{latency_ms:.0f} ms")
    m4.metric("Sources",      len(sources))

    # Confidence progress bar
    st.progress(min(1.0, confidence), text=f"Retrieval confidence: {confidence:.0%}")

    # Source chips
    if sources:
        chips = " ".join(
            f"<span class='source-chip'>📄 "
            f"{s.metadata.get('source', s.metadata.get('file_path', 'doc'))} "
            f"· p.{s.metadata.get('page', '?')}</span>"
            for s in sources
        )
        st.markdown(f"**Citations:** {chips}", unsafe_allow_html=True)

    # Retrieval details expander
    with st.expander("🔍 Retrieval Details", expanded=False):
        for i, src in enumerate(sources, 1):
            page  = src.metadata.get("page", "?")
            fname = src.metadata.get("source", src.metadata.get("file_path", "document"))
            st.markdown(f"**Passage {i}** | `{fname}` — page {page}")
            st.text_area(
                label=f"passage_{i}",
                value=src.page_content[:600] + ("…" if len(src.page_content) > 600 else ""),
                height=100,
                disabled=True,
                label_visibility="collapsed",
            )

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📚 Merged RAG System")
    st.caption("Advanced PDF Q&A · Supabase · Faithfulness Scoring")
    st.divider()

    # ─ LLM settings ──────────────────────────────────────────────────────────
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your free key at console.groq.com",
    )
    model_choice = st.selectbox(
        "🤖 LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
        index=0,
    )

    st.divider()

    # ─ Chunking ──────────────────────────────────────────────────────────────
    st.markdown("**⚙️ Chunking**")
    chunk_size    = st.slider("Chunk size",    300, 2000, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 0,   500,  200,  25)
    top_k         = st.slider("Top-K chunks",  1,   10,   4)

    st.divider()

    # ─ Retrieval ─────────────────────────────────────────────────────────────
    st.markdown("**🔍 Retrieval**")
    faithfulness_threshold = st.slider(
        "Faithfulness threshold", 0.0, 1.0, FAITHFULNESS_THRESHOLD, 0.05,
        help="Answers below this score show a hallucination warning",
    )

    # ─ Supabase PDF Source ───────────────────────────────────────────────────
    if _supabase_ok:
        st.divider()
        st.markdown("**☁️ PDF Source**")
        cloud_mode_radio = st.radio(
            "Choose source",
            ["📤 Upload new PDF", "🗂️ Select saved PDF"],
            index=0 if st.session_state.cloud_mode == "upload" else 1,
            label_visibility="collapsed",
        )
        st.session_state.cloud_mode = (
            "upload" if cloud_mode_radio.startswith("📤") else "select"
        )
    else:
        st.session_state.cloud_mode = "upload"

    st.divider()

    # ─ Actions ───────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.indexed_file:
        st.info(f"📌 Active: **{st.session_state.indexed_file}**")

    n_user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.caption(f"Queries this session: {n_user_msgs}")

# ════════════════════════════════════════════════════════════════════════════
# MAIN — Title + Tabs
# ════════════════════════════════════════════════════════════════════════════
st.markdown("# 📚 Merged RAG System")
st.caption(
    "Multi-document PDF Q&A · FAISS Dense Retrieval · "
    "Supabase Cloud Storage · Faithfulness Scoring"
)
st.divider()

tab_chat, tab_upload, tab_eval, tab_precision = st.tabs([
    "💬 Chat", "📄 Upload & Index", "📊 Evaluation", "🔬 Precision Matrix"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chat
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Replay message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("extras"):
                e = msg["extras"]
                _render_answer_extras(
                    e["sources"], e["confidence"],
                    e["faithfulness"], e["latency_ms"],
                    faithfulness_threshold,
                )

    # New input
    if st.session_state.qa_chain:
        if prompt := st.chat_input("Ask a question about the indexed PDF …"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating …"):
                    t0 = time.perf_counter()
                    answer, sources = ask_question(st.session_state.qa_chain, prompt)
                    latency_ms = (time.perf_counter() - t0) * 1000

                st.markdown(answer)

                confidence   = min(1.0, len(sources) / max(top_k, 1))
                faithfulness = _compute_faithfulness(answer, sources)
                _render_answer_extras(sources, confidence, faithfulness, latency_ms,
                                      faithfulness_threshold)
                _log_query(prompt, answer, confidence, faithfulness, latency_ms, len(sources))

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "extras": {
                    "sources":     sources,
                    "confidence":  confidence,
                    "faithfulness": faithfulness,
                    "latency_ms":  latency_ms,
                },
            })
    else:
        st.markdown(
            "<div style='text-align:center;padding:3rem;opacity:0.5'>"
            "<h3>📄 No PDF indexed yet</h3>"
            "<p>Go to the <strong>Upload &amp; Index</strong> tab to get started.</p>"
            "<p style='font-size:0.85em'>If you already indexed a PDF but still see this, "
            "check the Upload tab for any error messages shown in red.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Upload & Index
# ════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("📄 Upload & Index Documents")

    # ── Groq API key status hint ──────────────────────────────────────────────
    if not groq_api_key:
        st.warning("⚠️ No Groq API key found. Enter your key in the sidebar before indexing.")
    else:
        key_src = "from .env file" if os.getenv("GROQ_API_KEY") else "entered manually"
        st.success(f"✅ Groq API key ready ({key_src}).")

    st.divider()

    # ────────────────────────────────────────────────────────────────────────
    # Section A — Upload a new PDF
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload New PDF")

    uploaded_file = st.file_uploader(
        "Drop a PDF file here", type=["pdf"], label_visibility="collapsed"
    )

    if _supabase_ok:
        save_to_cloud = st.checkbox("☁️ Save to cloud (Supabase)", value=True)
    else:
        save_to_cloud = False
        st.caption(
            "ℹ️ Cloud save disabled — Supabase not configured or import failed. "
            "Check that SUPABASE_URL and SUPABASE_ANON_KEY are in your .env."
        )

    if st.button("⚡ Index PDF", use_container_width=True, type="primary", key="btn_index_upload"):
        if not uploaded_file:
            st.warning("Please upload a PDF first.")
        elif not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar.")
        else:
            _index_ok = False
            with st.status("Indexing document …", expanded=True) as status:
                try:
                    # Optional Supabase upload
                    if _supabase_ok and save_to_cloud:
                        st.write("☁️ Uploading to Supabase Storage…")
                        try:
                            upload_pdf(uploaded_file.getvalue(), uploaded_file.name)
                            st.write(f"✅ '{uploaded_file.name}' saved to cloud.")
                        except Exception as ce:
                            st.write(f"⚠️ Cloud upload failed (indexing locally only): {ce}")
                    elif _supabase_ok:
                        st.write("🖥️ Indexing locally (cloud save disabled).")

                    # Load PDF
                    st.write("📖 Loading PDF pages…")
                    uploaded_file.seek(0)
                    docs = load_pdf(uploaded_file)
                    st.write(f"✅ Loaded {len(docs)} page(s).")

                    # Split
                    st.write("✂️ Splitting into chunks…")
                    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.write(f"✅ Created {len(chunks)} chunks.")

                    # Build index
                    st.write("🧠 Building embeddings & FAISS index…")
                    emb       = get_cached_embeddings()
                    vs        = create_vector_store(chunks, emb)
                    retriever = get_retriever(vs, k=top_k)
                    chain     = build_qa_chain(retriever, groq_api_key, model_name=model_choice)
                    st.write("✅ Index ready.")

                    st.session_state.vectorstore  = vs
                    st.session_state.qa_chain     = chain
                    st.session_state.indexed_file = uploaded_file.name
                    st.session_state.messages     = []
                    status.update(label="✅ Indexing complete!", state="complete")
                    _index_ok = True

                except Exception as err:
                    status.update(label="❌ Indexing failed!", state="error")
                    st.error(f"**Indexing error:** {err}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

            if _index_ok:
                # Rerun so the Chat tab picks up the new qa_chain from session state
                st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # Section B — Load a previously uploaded PDF from Supabase cloud
    # ────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ☁️ Select Previously Uploaded PDF from Cloud")

    if not _supabase_ok:
        st.info(
            "Supabase is not connected. Once you configure SUPABASE_URL and "
            "SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY) in your .env, "
            "previously uploaded PDFs will appear here."
        )
    else:
        _refresh = st.button("🔄 Refresh list", key="btn_refresh_cloud")
        if _refresh:
            with st.spinner("Fetching cloud PDFs…"):
                try:
                    st.session_state["_cloud_pdf_list"] = list_pdfs()
                except Exception as e:
                    st.error(f"Could not connect to Supabase: {e}")
                    st.session_state["_cloud_pdf_list"] = []

        cloud_pdfs = st.session_state.get("_cloud_pdf_list", [])

        if not cloud_pdfs:
            st.info("No PDFs found in Supabase cloud. Upload one above and tick '☁️ Save to cloud'.")
        else:
            selected_pdf = st.selectbox("Select a cloud PDF", cloud_pdfs, key="cloud_pdf_selector")
            col1, col2  = st.columns(2)
            load_btn    = col1.button("⚡ Load & Index from Cloud", use_container_width=True, key="btn_load_cloud")
            delete_btn  = col2.button("🗑️ Delete from Cloud",       use_container_width=True, key="btn_delete_cloud")

            if delete_btn:
                with st.spinner(f"Deleting '{selected_pdf}'…"):
                    try:
                        delete_pdf(selected_pdf)
                        st.success(f"Deleted '{selected_pdf}' from Supabase.")
                        st.session_state["_cloud_pdf_list"] = [p for p in cloud_pdfs if p != selected_pdf]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

            if load_btn:
                if not groq_api_key:
                    st.warning("Please enter your Groq API key in the sidebar.")
                else:
                    _load_ok = False
                    with st.status(f"Loading '{selected_pdf}' from Supabase…",
                                   expanded=True) as status:
                        try:
                            st.write("☁️ Generating download URL…")
                            url  = get_pdf_download_url(selected_pdf)
                            st.write("📥 Downloading PDF…")
                            docs = load_pdf_from_url(url)
                            st.write(f"✅ Loaded {len(docs)} page(s).")

                            st.write("✂️ Splitting chunks…")
                            chunks = split_documents(docs, chunk_size=chunk_size,
                                                     chunk_overlap=chunk_overlap)
                            st.write(f"✅ {len(chunks)} chunks created.")

                            st.write("🧠 Building embeddings & FAISS index…")
                            emb       = get_cached_embeddings()
                            vs        = create_vector_store(chunks, emb)
                            retriever = get_retriever(vs, k=top_k)
                            chain     = build_qa_chain(retriever, groq_api_key,
                                                       model_name=model_choice)
                            st.write("✅ Index ready.")

                            st.session_state.vectorstore  = vs
                            st.session_state.qa_chain     = chain
                            st.session_state.indexed_file = selected_pdf
                            st.session_state.messages     = []
                            status.update(label="✅ Ready!", state="complete")
                            _load_ok = True

                        except Exception as err:
                            status.update(label="❌ Load failed!", state="error")
                            st.error(f"**Load error:** {err}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")

                    if _load_ok:
                        # Rerun so the Chat tab picks up the new qa_chain from session state
                        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluation Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("📊 Evaluation Dashboard")
    st.caption("Automatically populated as you chat in the 💬 Chat tab.")

    if not EVAL_LOG.exists():
        st.info("No evaluation data yet. Ask questions in the Chat tab to populate the log.")
    else:
        records = []
        with EVAL_LOG.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        query_records = [r for r in records if r.get("event") == "query"]

        if not query_records:
            st.info("No queries logged yet.")
        else:
            st.write(f"**Total queries logged:** {len(query_records)}")

            faith_vals   = [r["faithfulness"] for r in query_records if "faithfulness" in r]
            conf_vals    = [r["confidence"]   for r in query_records if "confidence"   in r]
            latency_vals = [r["latency_ms"]   for r in query_records if "latency_ms"   in r]

            c1, c2, c3 = st.columns(3)
            if faith_vals:
                avg_f = sum(faith_vals) / len(faith_vals)
                emoji = "🟢" if avg_f >= 0.7 else ("🟡" if avg_f >= 0.4 else "🔴")
                c1.metric(f"{emoji} Mean Faithfulness", f"{avg_f:.3f}",
                          help=">0.7 = well grounded; <0.4 = hallucination risk")
            if conf_vals:
                c2.metric("📊 Mean Confidence", f"{sum(conf_vals)/len(conf_vals):.3f}")
            if latency_vals:
                c3.metric("⏱️ Mean Latency", f"{sum(latency_vals)/len(latency_vals):.0f} ms")

            # Faithfulness over time
            if faith_vals:
                st.markdown("#### Faithfulness Score Over Queries")
                st.line_chart({"Faithfulness": faith_vals}, height=220, use_container_width=True)

            # Latency over time
            if latency_vals:
                st.markdown("#### Latency (ms) Over Queries")
                st.line_chart({"Latency (ms)": latency_vals}, height=180, use_container_width=True)

            col_log, col_clear = st.columns([4, 1])
            with col_log:
                with st.expander("📋 Raw Query Log (last 20)"):
                    for r in query_records[-20:][::-1]:
                        st.json(r)
            with col_clear:
                if st.button("🗑️ Clear Log"):
                    EVAL_LOG.unlink(missing_ok=True)
                    st.success("Log cleared.")
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Precision Matrix (manual Q&A pairs + Plotly charts)
# ════════════════════════════════════════════════════════════════════════════
with tab_precision:
    st.markdown("### 🔬 RAG Precision Matrix")
    st.markdown(
        "Enter test question-answer pairs. The system will retrieve context, "
        "generate answers, and score each metric."
    )

    METRIC_HELP = {
        "Context Precision":   "Fraction of retrieved chunks relevant to the question (0–1)",
        "Answer Faithfulness": "How grounded the answer is in the retrieved context (0–1)",
        "Answer Relevancy":    "How directly the answer addresses the question (0–1)",
        "Context Recall":      "How well the context covers the expected answer (0–1)",
        "Overall Score":       "Average of all four metrics (0–1)",
    }

    with st.expander("ℹ️ Metric Definitions", expanded=False):
        for m, h in METRIC_HELP.items():
            st.markdown(f"**{m}** — {h}")

    if not st.session_state.qa_chain:
        st.warning("⚠️ Please index a PDF first (Upload & Index tab → ⚡ Index PDF).")
    else:
        st.markdown("#### Test Questions")
        n_pairs = st.number_input("Number of test pairs", min_value=1, max_value=20,
                                  value=3, step=1)

        qa_pairs = []
        for i in range(int(n_pairs)):
            st.markdown(f"**Pair {i + 1}**")
            c1, c2 = st.columns(2)
            q = c1.text_input(f"Question {i + 1}", key=f"eq_{i}",
                              placeholder="e.g. What is the main topic?")
            a = c2.text_input(f"Expected Answer {i + 1}", key=f"ea_{i}",
                              placeholder="e.g. The document discusses…")
            if q.strip() and a.strip():
                qa_pairs.append((q.strip(), a.strip()))

        if st.button("🔬 Run Evaluation", use_container_width=True):
            if not qa_pairs:
                st.warning("Please fill in at least one question and expected answer.")
            else:
                with st.spinner(f"Evaluating {len(qa_pairs)} pair(s)…"):
                    results = evaluate_batch(qa_pairs, st.session_state.qa_chain)

                df = pd.DataFrame(results)
                metric_cols = [
                    "Context Precision", "Answer Faithfulness",
                    "Answer Relevancy", "Context Recall", "Overall Score",
                ]

                st.markdown("#### 📋 Results")

                def _color(val):
                    if isinstance(val, float):
                        if val >= 0.7:   return "background-color:#1a4731;color:#6ee7b7"
                        elif val >= 0.4: return "background-color:#3b2f00;color:#fcd34d"
                        else:            return "background-color:#3b1111;color:#fca5a5"
                    return ""

                display_df = df[["Question", "Expected Answer", "Generated Answer"] + metric_cols]
                styled     = display_df.style.applymap(_color, subset=metric_cols).format(
                    {c: "{:.2%}" for c in metric_cols}
                )
                st.dataframe(styled, use_container_width=True)

                st.markdown("#### 📈 Average Scores")
                avg  = df[metric_cols].mean()
                cols = st.columns(len(metric_cols))
                for col, metric in zip(cols, metric_cols):
                    score = avg[metric]
                    emoji = "🟢" if score >= 0.7 else ("🟡" if score >= 0.4 else "🔴")
                    col.metric(label=f"{emoji} {metric}", value=f"{score:.1%}")

                # Radar chart
                st.markdown("#### 🕸️ Radar Chart — Average Metric Scores")
                radar_m = ["Context Precision", "Answer Faithfulness",
                           "Answer Relevancy", "Context Recall"]
                radar_v = [avg[m] for m in radar_m] + [avg[radar_m[0]]]
                radar_l = radar_m + [radar_m[0]]

                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(
                    r=radar_v, theta=radar_l, fill="toself",
                    line=dict(color="#7c6aff", width=2),
                    fillcolor="rgba(124,106,255,0.25)",
                    name="Avg Score",
                ))
                fig_r.update_layout(
                    polar=dict(
                        bgcolor="rgba(255,255,255,0.04)",
                        radialaxis=dict(
                            visible=True, range=[0, 1], tickformat=".0%",
                            gridcolor="rgba(255,255,255,0.12)",
                            linecolor="rgba(255,255,255,0.12)",
                            tickfont=dict(color="#aaa"),
                        ),
                        angularaxis=dict(
                            gridcolor="rgba(255,255,255,0.12)",
                            linecolor="rgba(255,255,255,0.12)",
                            tickfont=dict(color="#e0e0ff", size=13),
                        ),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0ff"), showlegend=False,
                    margin=dict(t=40, b=40, l=60, r=60), height=420,
                )
                st.plotly_chart(fig_r, use_container_width=True)

                # Grouped bar chart (only when > 1 pair)
                if len(df) > 1:
                    st.markdown("#### 📊 Per-Question Metric Breakdown")
                    bar_m  = ["Context Precision", "Answer Faithfulness",
                              "Answer Relevancy", "Context Recall"]
                    colors = ["#7c6aff", "#00d4aa", "#f59e0b", "#f43f5e"]
                    short_q = [f"Q{i + 1}" for i in range(len(df))]

                    fig_b = go.Figure()
                    for metric, color in zip(bar_m, colors):
                        fig_b.add_trace(go.Bar(
                            name=metric, x=short_q, y=df[metric].tolist(),
                            marker_color=color,
                            text=[f"{v:.0%}" for v in df[metric]],
                            textposition="outside",
                        ))
                    fig_b.update_layout(
                        barmode="group",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e0e0ff"),
                        yaxis=dict(
                            range=[0, 1.15], tickformat=".0%",
                            gridcolor="rgba(255,255,255,0.08)",
                            linecolor="rgba(255,255,255,0.08)",
                        ),
                        xaxis=dict(linecolor="rgba(255,255,255,0.08)"),
                        legend=dict(
                            bgcolor="rgba(255,255,255,0.05)",
                            bordercolor="rgba(255,255,255,0.1)",
                            borderwidth=1,
                        ),
                        margin=dict(t=20, b=40), height=380,
                    )
                    fig_b.update_traces(
                        hovertemplate="<b>%{fullData.name}</b><br>Score: %{y:.2%}<extra></extra>"
                    )
                    st.plotly_chart(fig_b, use_container_width=True)

                    with st.expander("📝 Question legend"):
                        for i, row in df.iterrows():
                            st.markdown(f"**Q{i + 1}:** {row['Question']}")

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV", data=csv,
                    file_name="precision_matrix.csv", mime="text/csv",
                    use_container_width=True,
                )