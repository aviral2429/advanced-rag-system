# 📚 PDF RAG Chatbot — Advanced Q&A System

> **Retrieval-Augmented Generation (RAG)** powered PDF question-answering system with Supabase cloud storage, faithfulness scoring, evaluation dashboard, and AI-generated mock tests.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green?logo=chainlink&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-Cloud%20Storage-3ECF8E?logo=supabase&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM%20API-orange?logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple?logoColor=white)

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Chat with PDFs** | Ask natural language questions — get grounded answers with source citations |
| ☁️ **Supabase Cloud Storage** | Upload PDFs once, access from anywhere via cloud backup |
| 🧪 **AI Mock Tests** | Auto-generate MCQs from any PDF — answer, submit, and get scored |
| 📊 **Evaluation Dashboard** | Track faithfulness, confidence, and latency over every query |
| 🔬 **Precision Matrix** | Radar chart & bar charts for Context Precision, Recall, Faithfulness, Relevancy |
| 🧠 **FAISS Dense Retrieval** | Local vector search with HuggingFace embeddings (no API needed) |
| ⚡ **Groq LLM** | Ultra-fast inference with `llama-3.3-70b-versatile` and other models |

---

## 🏗️ Architecture

```
PDF Upload
    │
    ├──► Supabase Storage (cloud backup)
    │
    ▼
PDF Loader (PyPDF)
    │
    ▼
Text Splitter (LangChain RecursiveCharacterTextSplitter)
    │
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2, runs locally)
    │
    ▼
FAISS Vector Store
    │
    ├──► RAG Chain (Groq LLM)  ──► Chat Tab
    │
    ├──► MCQ Generator (Groq)  ──► Mock Test Tab
    │
    └──► Evaluation Engine     ──► Eval + Precision Tabs
```

---

## 🖥️ App Tabs

### 💬 Chat
- Ask questions about any indexed PDF
- Answers grounded strictly in document context
- Per-answer metrics: **Confidence**, **Faithfulness**, **Latency**, **Sources**
- Source passage viewer with page numbers
- Hallucination warning when faithfulness < threshold

### 📄 Upload & Index
- **Upload new PDF** → auto-backed up to Supabase cloud
- **Select from cloud** → load previously uploaded PDFs by name
- Delete cloud PDFs directly from the UI

### 🧪 Mock Test
- Choose number of questions (5/10/15/20) and difficulty (Easy/Medium/Hard)
- Groq LLM generates MCQs with 4 options + explanations — all from the PDF content
- Interactive radio button quiz
- After submission: color-coded review (✅ correct / ❌ wrong / ☑️ correct answer)
- Score card with grade: Excellent 🏆 / Good 👍 / Needs improvement 📚 / Keep studying 💪
- Retake button to reset and try again

### 📊 Evaluation Dashboard
- Automatically logs every query with faithfulness, confidence, latency
- Live line charts over time
- Mean metrics at a glance

### 🔬 Precision Matrix
- Enter custom Q&A test pairs
- Metrics: Context Precision, Answer Faithfulness, Answer Relevancy, Context Recall
- Radar chart + grouped bar chart
- Download results as CSV

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/aviral2429/advanced-rag-system.git
cd advanced-rag-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
# Groq API — get free key at https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Supabase — get from https://supabase.com/dashboard → Project Settings → API
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-public-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
```

### 4. Set up Supabase Storage

1. Go to your [Supabase Dashboard](https://supabase.com/dashboard)
2. Create a **Storage bucket** named exactly `pdfs` and set it to **Public**
3. Open **SQL Editor** and run the contents of [`setup_supabase_policies.sql`](./setup_supabase_policies.sql)

### 5. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser 🎉

---

## 📁 Project Structure

```
advanced-rag-system/
├── app.py                      # Main Streamlit app (all tabs & UI)
├── cloud_storage.py            # Supabase Storage integration
├── pdf_loader.py               # PDF loading (file upload + URL)
├── text_splitter.py            # LangChain document chunking
├── embeddings.py               # HuggingFace local embeddings
├── vector_store.py             # FAISS vector store creation
├── qa_system.py                # Groq RAG chain (LCEL)
├── evaluation.py               # Precision matrix evaluation engine
├── setup_supabase_policies.sql # RLS policies for Supabase bucket
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── .gitignore                  # Excludes .env, __pycache__, eval logs
```

---

## 🔑 API Keys Needed

| Service | Purpose | Free Tier |
|---|---|---|
| [Groq](https://console.groq.com) | LLM inference (chat + MCQ generation) | ✅ Yes |
| [Supabase](https://supabase.com) | PDF cloud storage | ✅ Yes (500MB) |

HuggingFace embeddings run **100% locally** — no API key needed.

---

## ⚙️ Configuration (Sidebar)

| Setting | Default | Description |
|---|---|---|
| Groq API Key | from `.env` | Your Groq key |
| LLM Model | `llama-3.3-70b-versatile` | Choose from 4 Groq models |
| Chunk size | 1000 | Characters per text chunk |
| Chunk overlap | 200 | Overlap between chunks |
| Top-K chunks | 4 | Retrieved passages per query |
| Faithfulness threshold | 0.50 | Below this → hallucination warning |

---

## 🧪 Mock Test Flow

```
Index PDF  →  Go to 🧪 Mock Test tab
           →  Select: 10 questions, Medium difficulty
           →  Click ✨ Generate New Test
           →  Answer all MCQs (radio buttons)
           →  Click 🏁 Submit Test
           →  See score + per-question review with explanations
           →  Click 🔄 Retake to try again
```

---

## 🛡️ Security Notes

- **Never commit your `.env` file** — it's in `.gitignore`
- Use `SUPABASE_SERVICE_ROLE_KEY` server-side only — it bypasses RLS
- The `SUPABASE_ANON_KEY` is safe for client-side use

---

## 📦 Dependencies

```
streamlit          ≥ 1.40   — Web UI framework
langchain          ≥ 0.3    — RAG orchestration
langchain-community≥ 0.3    — Community integrations
langchain-groq     ≥ 0.2    — Groq LLM integration
faiss-cpu          ≥ 1.8    — Vector similarity search
sentence-transformers≥ 2.7  — Local HuggingFace embeddings
pypdf              ≥ 3.4    — PDF parsing
supabase           ≥ 2.0    — Cloud storage client
python-dotenv      ≥ 1.0    — .env loading
plotly             ≥ 5.18   — Interactive charts
pandas             ≥ 2.0    — Data handling
numpy              ≥ 1.26   — Numerical operations
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push and open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Made with ❤️ using Streamlit · LangChain · Groq · Supabase · FAISS
</div>
