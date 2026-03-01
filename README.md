<<<<<<< HEAD
# 📚 Advanced RAG System

A research-grade **Retrieval-Augmented Generation (RAG)** system for multi-document PDF question answering, with hybrid retrieval, cross-encoder reranking, and faithfulness-guided hallucination mitigation.

---

## ✨ Features

- **Hybrid Retrieval** — FAISS dense search + BM25 sparse search fused with Reciprocal Rank Fusion (RRF)
- **Semantic Chunking** — Embedding-guided sentence-level breakpoint detection (NLTK)
- **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-12-v2` for precision re-scoring
- **Faithfulness Scoring** — Reference-free hallucination detection via cosine similarity
- **Multi-Provider LLM** — Groq, OpenAI, Anthropic, Together.ai (swap with one env var)
- **Streamlit UI** — Chat, Upload & Index, Evaluation dashboard in one app
- **Index Persistence** — Save/load FAISS + BM25 index across sessions
- **Evaluation Metrics** — Recall@K, MRR, Precision@K logged to JSONL

---

## 🗂️ Project Structure

```
advanced-rag-system/
├── app.py                  # Streamlit web UI
├── config.py               # Pydantic-validated configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── diagnose_index.py       # Index debugging utility
└── src/
    ├── pipeline.py         # End-to-end RAG orchestration
    ├── hybrid_retriever.py # FAISS + BM25 + RRF + CrossEncoder
    ├── embeddings.py       # BAAI/bge-small-en-v1.5 bi-encoder
    ├── chunker.py          # Semantic + recursive chunking
    ├── loader.py           # PDF loader (pdfplumber + pypdf)
    ├── llm.py              # Multi-provider LLM client
    └── evaluation.py       # Metrics: Recall@K, MRR, Faithfulness
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/aviral2429/advanced-rag-system.git
cd advanced-rag-system
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and fill in your API key:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔧 Configuration

All settings are controlled via `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | Provider: `groq` / `openai` / `anthropic` / `together` |
| `GROQ_API_KEY` | — | Groq API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `TOGETHER_API_KEY` | — | Together.ai API key |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model name for Groq |
| `CHUNK_SIZE` | `1024` | Characters per chunk |
| `USE_RERANKER` | `true` | Enable cross-encoder reranker |
| `TOP_K` | `5` | Top-K chunks to retrieve |

---

## 🛠️ Tech Stack

| Layer | Tool | Version |
|---|---|---|
| UI | Streamlit | 1.41.1 |
| Embeddings | BAAI/bge-small-en-v1.5 + SentenceTransformers | 3.4.1 |
| Dense Index | FAISS (faiss-cpu) | 1.8.0 |
| Sparse Index | BM25Okapi (rank-bm25) | 0.2.2 |
| Reranker | ms-marco-MiniLM-L-12-v2 | — |
| LLM APIs | Groq / OpenAI / Anthropic / Together.ai | — |
| PDF Parsing | pdfplumber + pypdf | 0.11.9 / 5.1.0 |
| Config | Pydantic-Settings | 2.7.0 |

---

## 📊 How It Works

```
PDF Upload → PDFLoader → SemanticChunker → EmbeddingModel
                                                  ↓
                                    FAISS Dense Index + BM25 Sparse Index
                                                  ↓
                              Query → Dense + Sparse → RRF Fusion
                                                  ↓
                                       CrossEncoder Reranker
                                                  ↓
                                    LLMClient (grounded prompt)
                                                  ↓
                              Answer + Citations + Faithfulness Score
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Aviral** — [@aviral2429](https://github.com/aviral2429)
=======
# advanced-rag-system
>>>>>>> d5b2c1cadb488298d9ded51a84a6a2b74b557290
