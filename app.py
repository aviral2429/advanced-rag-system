import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

from pdf_loader import load_pdf, load_pdf_from_url
from text_splitter import split_documents
from embeddings import get_embeddings
from vector_store import create_vector_store, get_retriever
from qa_system import build_qa_chain, ask_question
from evaluation import evaluate_batch

load_dotenv()

# ── Supabase availability check ───────────────────────────────────────────────
try:
    from cloud_storage import upload_pdf, list_pdfs, get_pdf_download_url, delete_pdf
    _firebase_configured = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY"))
except ImportError:
    _firebase_configured = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
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

    /* Cards */
    .chat-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        animation: fadeUp 0.35s ease both;
    }
    .user-card  { border-left: 3px solid #7c6aff; }
    .bot-card   { border-left: 3px solid #00d4aa; }

    .source-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-top: 0.4rem;
        font-size: 0.82rem;
        color: #aaa !important;
    }

    @keyframes fadeUp {
        from { opacity:0; transform:translateY(12px); }
        to   { opacity:1; transform:translateY(0);    }
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

    /* Input */
    .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    /* Headings */
    h1, h2, h3, p, .stMarkdown { color: #e0e0ff !important; }
    .stAlert { border-radius: 10px !important; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
for key, val in {
    "messages": [],
    "vectorstore": None,
    "qa_chain": None,
    "indexed_file": None,
    "cloud_mode": "upload",   # 'upload' or 'select'
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF RAG Chatbot")
    st.markdown("Upload a PDF and chat with it using AI.")
    st.divider()

    # Groq API key
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your free key at console.groq.com",
    )

    # Model selector
    model_choice = st.selectbox(
        "🤖 LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
        index=0,
    )

    st.divider()

    # ── Chunking config ───────────────────────────────────────────────────────
    chunk_size    = st.slider("Chunk size",    300, 2000, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 0,   500,  200, 25)
    top_k         = st.slider("Top-K results", 1,   10,   4)

    st.divider()

    # ── PDF source ────────────────────────────────────────────────────────────
    if _firebase_configured:
        st.markdown("### ☁️ PDF Source")
        cloud_mode = st.radio(
            "Choose source",
            ["📤 Upload new PDF", "🗂️ Select saved PDF"],
            index=0 if st.session_state.cloud_mode == "upload" else 1,
            label_visibility="collapsed",
        )
        st.session_state.cloud_mode = "upload" if cloud_mode.startswith("📤") else "select"
    else:
        st.session_state.cloud_mode = "upload"

    # ── Upload new PDF ────────────────────────────────────────────────────────
    if st.session_state.cloud_mode == "upload":
        uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

        # Cloud save toggle (only shown when Supabase is configured)
        if _firebase_configured:
            save_to_cloud = st.checkbox(
                "☁️ Save to cloud (Supabase)",
                value=True,
                help="If checked, the PDF will be saved to Supabase so you can reload it anytime without re-uploading.",
            )
        else:
            save_to_cloud = False

        index_btn = st.button("⚡ Index PDF", use_container_width=True)

        if index_btn:
            if not uploaded_file:
                st.warning("Please upload a PDF first.")
            elif not groq_api_key:
                st.warning("Please enter your Groq API key.")
            else:
                # ─ Save to Supabase cloud (optional) ──────────────────────────────
                if _firebase_configured and save_to_cloud:
                    with st.spinner("☁️ Uploading to Supabase Storage…"):
                        try:
                            file_bytes = uploaded_file.getvalue()
                            upload_pdf(file_bytes, uploaded_file.name)
                            st.success(f"☁️ '{uploaded_file.name}' saved to cloud — accessible anytime.")
                        except Exception as e:
                            st.warning(f"⚠️ Cloud upload failed (indexing locally only): {e}")
                elif _firebase_configured and not save_to_cloud:
                    st.info("🖥️ Indexing locally only — PDF will not be saved to cloud.")

                # ─ Load & index ───────────────────────────────────────────────
                with st.spinner("Loading PDF…"):
                    uploaded_file.seek(0)
                    docs = load_pdf(uploaded_file)
                st.success(f"Loaded {len(docs)} page(s).")

                with st.spinner("Splitting into chunks…"):
                    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.success(f"Created {len(chunks)} chunks.")

                with st.spinner("Creating embeddings & index…"):
                    emb = get_embeddings()
                    vs  = create_vector_store(chunks, emb)
                    retriever = get_retriever(vs, k=top_k)
                    chain = build_qa_chain(retriever, groq_api_key, model_name=model_choice)

                st.session_state.vectorstore   = vs
                st.session_state.qa_chain      = chain
                st.session_state.indexed_file  = uploaded_file.name
                st.session_state.messages      = []
                st.success(f"✅ '{uploaded_file.name}' indexed! Start chatting →")

    # ── Select saved PDF from Firebase ────────────────────────────────────────
    else:
        with st.spinner("Fetching cloud PDFs…"):
            try:
                cloud_pdfs = list_pdfs()
            except Exception as e:
                st.error(f"Could not connect to Firebase: {e}")
                cloud_pdfs = []

        if not cloud_pdfs:
            st.info("No PDFs found in cloud storage. Upload one first.")
        else:
            selected_pdf = st.selectbox("Select a PDF", cloud_pdfs)
            col1, col2 = st.columns(2)
            load_btn   = col1.button("⚡ Load & Index", use_container_width=True)
            delete_btn = col2.button("🗑️ Delete", use_container_width=True)

            if delete_btn:
                with st.spinner(f"Deleting '{selected_pdf}'…"):
                    try:
                        delete_pdf(selected_pdf)
                        st.success(f"Deleted '{selected_pdf}' from cloud.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

            if load_btn:
                if not groq_api_key:
                    st.warning("Please enter your Groq API key.")
                else:
                    with st.spinner(f"☁️ Downloading '{selected_pdf}' from Firebase…"):
                        try:
                            url  = get_pdf_download_url(selected_pdf)
                            docs = load_pdf_from_url(url)
                        except Exception as e:
                            st.error(f"Failed to fetch from Firebase: {e}")
                            docs = []

                    if docs:
                        st.success(f"Loaded {len(docs)} page(s).")

                        with st.spinner("Splitting into chunks…"):
                            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        st.success(f"Created {len(chunks)} chunks.")

                        with st.spinner("Creating embeddings & index…"):
                            emb = get_embeddings()
                            vs  = create_vector_store(chunks, emb)
                            retriever = get_retriever(vs, k=top_k)
                            chain = build_qa_chain(retriever, groq_api_key, model_name=model_choice)

                        st.session_state.vectorstore   = vs
                        st.session_state.qa_chain      = chain
                        st.session_state.indexed_file  = selected_pdf
                        st.session_state.messages      = []
                        st.success(f"✅ '{selected_pdf}' indexed! Start chatting →")

    if st.session_state.indexed_file:
        st.info(f"📌 Active: **{st.session_state.indexed_file}**")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main area — Tabs ─────────────────────────────────────────────────────────
st.markdown("# 🤖 PDF RAG Chatbot")
if st.session_state.indexed_file:
    st.markdown(f"Chatting with **{st.session_state.indexed_file}**")
else:
    st.markdown("Upload and index a PDF from the sidebar to get started.")

st.divider()

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Precision Matrix"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-card user-card">👤 <strong>You</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-card bot-card">🤖 <strong>Assistant</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                with st.expander("📚 Source excerpts"):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.metadata.get("page", "?")
                        snippet = src.page_content[:300].replace("\n", " ")
                        st.markdown(
                            f'<div class="source-card"><strong>Source {i} — Page {page}</strong><br>{snippet}…</div>',
                            unsafe_allow_html=True,
                        )

    if st.session_state.qa_chain:
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask a question about the PDF",
                placeholder="e.g. What is the main topic of this document?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking…"):
                answer, sources = ask_question(st.session_state.qa_chain, user_input.strip())
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.rerun()
    else:
        st.markdown(
            """
            <div style='text-align:center; padding: 3rem; opacity:0.5;'>
                <h3>📄 No PDF indexed yet</h3>
                <p>Use the sidebar to upload and index a PDF document.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Tab 2: Precision Matrix ───────────────────────────────────────────────────
with tab_eval:
    st.markdown("### 📊 RAG Precision Matrix")
    st.markdown(
        "Enter test question-answer pairs below. The system will retrieve context, "
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
        st.warning("⚠️ Please index a PDF first (sidebar → ⚡ Index PDF).")
    else:
        st.markdown("#### Test Questions")
        n_pairs = st.number_input("Number of test pairs", min_value=1, max_value=20, value=3, step=1)

        qa_pairs = []
        for i in range(int(n_pairs)):
            st.markdown(f"**Pair {i+1}**")
            c1, c2 = st.columns(2)
            with c1:
                q = st.text_input(f"Question {i+1}", key=f"eq_{i}", placeholder="e.g. What is the main topic?")
            with c2:
                a = st.text_input(f"Expected Answer {i+1}", key=f"ea_{i}", placeholder="e.g. The document discusses…")
            if q.strip() and a.strip():
                qa_pairs.append((q.strip(), a.strip()))

        run_eval = st.button("🔬 Run Evaluation", use_container_width=True)

        if run_eval:
            if not qa_pairs:
                st.warning("Please fill in at least one question and expected answer.")
            else:
                with st.spinner(f"Evaluating {len(qa_pairs)} pair(s)…"):
                    results = evaluate_batch(qa_pairs, st.session_state.qa_chain)

                df = pd.DataFrame(results)
                metric_cols = ["Context Precision", "Answer Faithfulness", "Answer Relevancy", "Context Recall", "Overall Score"]

                st.markdown("#### 📋 Results")

                # Color-coded metric table
                def _color(val):
                    if isinstance(val, float):
                        if val >= 0.7:   return "background-color: #1a4731; color: #6ee7b7"
                        elif val >= 0.4: return "background-color: #3b2f00; color: #fcd34d"
                        else:            return "background-color: #3b1111; color: #fca5a5"
                    return ""

                display_df = df[["Question", "Expected Answer", "Generated Answer"] + metric_cols]
                styled = display_df.style.applymap(_color, subset=metric_cols).format(
                    {c: "{:.2%}" for c in metric_cols}
                )
                st.dataframe(styled, use_container_width=True)

                # Averages — metric cards
                st.markdown("#### 📈 Average Scores")
                avg = df[metric_cols].mean()
                cols = st.columns(len(metric_cols))
                for col, metric in zip(cols, metric_cols):
                    score = avg[metric]
                    emoji = "🟢" if score >= 0.7 else ("🟡" if score >= 0.4 else "🔴")
                    col.metric(label=f"{emoji} {metric}", value=f"{score:.1%}")

                # ── Chart 1: Radar chart — average across all metrics ──────────
                st.markdown("#### 🕸️ Radar Chart — Average Metric Scores")
                radar_metrics = ["Context Precision", "Answer Faithfulness",
                                 "Answer Relevancy", "Context Recall"]
                radar_vals = [avg[m] for m in radar_metrics]
                radar_vals_closed = radar_vals + [radar_vals[0]]   # close the polygon
                radar_labels_closed = radar_metrics + [radar_metrics[0]]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_vals_closed,
                    theta=radar_labels_closed,
                    fill="toself",
                    line=dict(color="#7c6aff", width=2),
                    fillcolor="rgba(124, 106, 255, 0.25)",
                    name="Avg Score",
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(255,255,255,0.04)",
                        radialaxis=dict(
                            visible=True, range=[0, 1],
                            tickformat=".0%",
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
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0ff"),
                    showlegend=False,
                    margin=dict(t=40, b=40, l=60, r=60),
                    height=420,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # ── Chart 2: Grouped bar — per question breakdown ─────────────
                if len(df) > 1:
                    st.markdown("#### 📊 Per-Question Metric Breakdown")
                    bar_metrics = ["Context Precision", "Answer Faithfulness",
                                   "Answer Relevancy", "Context Recall"]
                    colors = ["#7c6aff", "#00d4aa", "#f59e0b", "#f43f5e"]
                    short_q = [f"Q{i+1}" for i in range(len(df))]

                    fig_bar = go.Figure()
                    for metric, color in zip(bar_metrics, colors):
                        fig_bar.add_trace(go.Bar(
                            name=metric,
                            x=short_q,
                            y=df[metric].tolist(),
                            marker_color=color,
                            text=[f"{v:.0%}" for v in df[metric]],
                            textposition="outside",
                        ))

                    fig_bar.update_layout(
                        barmode="group",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
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
                        margin=dict(t=20, b=40),
                        height=380,
                    )
                    # Hover tooltip shows full question text
                    fig_bar.update_traces(
                        hovertemplate="<b>%{fullData.name}</b><br>Score: %{y:.2%}<extra></extra>"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Question legend below chart
                    with st.expander("📝 Question legend"):
                        for i, row in df.iterrows():
                            st.markdown(f"**Q{i+1}:** {row['Question']}")

                # Download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv,
                    file_name="precision_matrix.csv",
                    mime="text/csv",
                    use_container_width=True,
                )