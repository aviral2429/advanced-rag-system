# 🚀 Advanced RAG System for PDF Intelligence

> **Chat with PDFs, generate AI mock tests, and evaluate answer quality with a production-style Retrieval-Augmented Generation (RAG) pipeline.**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/RAG-LangChain-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-purple)
![Supabase](https://img.shields.io/badge/Storage-Supabase-3ECF8E?logo=supabase&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-black)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

---

## 🧩 Project Overview

**Advanced RAG System** is a portfolio-grade AI application that combines document retrieval, LLM reasoning, and evaluation analytics in one clean interface.

It allows you to:
- upload and index PDF documents,
- ask grounded questions with citations,
- generate MCQ-style mock tests from source content,
- monitor faithfulness, confidence, and latency metrics over time.

This repository is designed to demonstrate practical AI engineering skills across **LLM integration, retrieval pipelines, vector search, cloud storage, and quality evaluation**.

---

## ✨ Core Features

| Area | Capability | Why It Matters |
|---|---|---|
| 💬 Document Q&A | Ask natural-language questions over indexed PDFs | Demonstrates end-to-end RAG query flow |
| 📌 Source Grounding | Returns retrieved context and source citations | Improves trust and explainability |
| ☁️ Cloud PDF Library | Upload, list, load, and delete PDFs via Supabase Storage | Enables persistent multi-session workflows |
| 🧪 AI Mock Test Generator | Creates MCQs with explanations from PDF content | Shows advanced LLM prompting use cases |
| 📊 Evaluation Dashboard | Tracks faithfulness, confidence, and response latency | Adds measurable quality visibility |
| 🔬 Precision Matrix | Scores Context Precision, Recall, Relevancy, Faithfulness | Demonstrates model/output evaluation rigor |
| ⚡ Local Semantic Retrieval | Uses FAISS + local sentence-transformer embeddings | Reduces API cost for retrieval layer |

---

## 🧱 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| UI | Streamlit |
| RAG Orchestration | LangChain (LCEL) |
| LLM Inference | Groq (Llama / Mixtral / Gemma models) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (`faiss-cpu`) |
| PDF Parsing | PyPDF (`pypdf`) |
| Cloud Storage | Supabase Storage |
| Data & Charts | pandas, numpy, plotly |

---

## 🏗️ High-Level Architecture

```text
PDF Upload / Cloud Selection
          ↓
      PDF Loader
          ↓
   Text Chunking
          ↓
 Local Embeddings
          ↓
    FAISS Index
          ↓
   Retriever + LLM
          ↓
 Chat Answers / MCQ Generation / Evaluation Metrics
```

---

## 📸 Screenshots / Demo (Placeholders)

> Replace these placeholders with real assets before public sharing.

- `docs/screenshots/chat-tab.png` — Chat interface with citations
- `docs/screenshots/upload-tab.png` — Upload + cloud management flow
- `docs/screenshots/mock-test-tab.png` — Generated MCQ test and scoring view
- `docs/screenshots/evaluation-dashboard.png` — Faithfulness/confidence/latency charts
- `docs/demo/app-walkthrough.gif` — 20–40 second workflow demo

Example markdown (update paths once assets are added):

```md
![Chat Tab](docs/screenshots/chat-tab.png)
![Demo GIF](docs/demo/app-walkthrough.gif)
```

---

## 🚀 Installation

### 1) Clone the repository

```bash
git clone https://github.com/aviral2429/advanced-rag-system.git
cd advanced-rag-system
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

```bash
cp .env.example .env
```

Populate `.env` with your credentials:

```env
GROQ_API_KEY=your_groq_api_key_here
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-public-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
```

### 4) Configure Supabase bucket

1. Create a storage bucket named **`pdfs`**.
2. Run SQL from `setup_supabase_policies.sql` in Supabase SQL Editor.

### 5) Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 Usage Guide

1. Go to **Upload & Index** and upload a PDF (optionally save to cloud).
2. Click **Index PDF** to build chunks, embeddings, and FAISS index.
3. Use **Chat** tab to ask questions and inspect citations.
4. Use **Mock Test** tab to generate and complete AI-created MCQs.
5. Use **Evaluation** and **Precision Matrix** tabs to analyze answer quality.

---

## 📁 Project Structure

```text
advanced-rag-system/
├── app.py                        # Streamlit application (UI + workflow orchestration)
├── cloud_storage.py              # Supabase storage integration
├── pdf_loader.py                 # PDF ingestion (file + URL)
├── text_splitter.py              # Chunking strategy
├── embeddings.py                 # Local embedding model loader
├── vector_store.py               # FAISS index/retriever utilities
├── qa_system.py                  # RAG chain construction and query execution
├── evaluation.py                 # Precision matrix evaluation metrics
├── setup_supabase_policies.sql   # Supabase storage policy setup
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # Repository documentation
```

---

## 🗺️ Future Enhancements

- Multi-document session memory and cross-document comparison
- Persistent vector index caching across restarts
- Role-based access controls for cloud documents
- Automated evaluation datasets and benchmark reports
- Optional API layer (FastAPI) for integration with external clients
- CI pipeline for linting, tests, and packaging checks

---

## 🤝 Contributing

Contributions are welcome and appreciated.

1. Fork the repository
2. Create a branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Commit with a clear message: `git commit -m "feat: add your feature"`
5. Push your branch and open a Pull Request

For high-quality contributions, include:
- a clear problem statement,
- before/after behavior,
- and screenshots for UI changes.

---

## 🔐 Security Notes

- Never commit `.env` or private keys.
- Treat `SUPABASE_SERVICE_ROLE_KEY` as sensitive server-side secret.
- Prefer restrictive Supabase policies for production environments.

---

## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for full text.

---

## 📌 GitHub About (Recommended)

**Description (within GitHub limit):**
Production-ready PDF RAG app with LangChain, FAISS, Groq, and Supabase for citation-grounded Q&A, AI mock tests, and retrieval quality evaluation.

**Tagline:**
Ask PDFs smarter with grounded RAG.

**Topics:**
`rag`, `retrieval-augmented-generation`, `llm`, `generative-ai`, `langchain`, `streamlit`, `faiss`, `vector-search`, `semantic-search`, `pdf-chatbot`, `document-question-answering`, `supabase`, `groq`, `sentence-transformers`, `huggingface`, `ai-evaluation`, `python`, `chatbot`, `edtech`, `machine-learning`

---

<div align="center">
Built for practical AI engineering portfolios • Streamlit × LangChain × FAISS × Groq × Supabase
</div>
