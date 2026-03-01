"""
api.py — FastAPI REST backend for the Advanced RAG System.

Exposes:
  POST /api/index          — Upload PDFs and index them
  POST /api/query          — Ask a question
  GET  /api/stats          — Index statistics
  GET  /api/evaluation     — Evaluation log metrics
  POST /api/save           — Persist index to disk
  POST /api/load           — Load index from disk
  GET  /api/health         — Health check
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

# Ensure project root on sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import RAGConfig
from src.pipeline import RAGPipeline

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Advanced RAG System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton pipeline ────────────────────────────────────────────────────────

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(RAGConfig())
    return _pipeline


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_reranker: bool = True
    temperature: float = 0.1


class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    faithfulness: float
    hallucination_warning: bool
    latency_ms: float
    retrieval_method: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline_ready": get_pipeline().is_indexed}


@app.get("/api/stats")
def stats():
    pipeline = get_pipeline()
    return {
        "is_indexed": pipeline.is_indexed,
        **pipeline.stats,
    }


@app.post("/api/index")
async def index_documents(files: List[UploadFile] = File(...)):
    """Accept multiple PDF files, index them and return stats."""
    pipeline = get_pipeline()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for uploaded in files:
            if not uploaded.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"{uploaded.filename} is not a PDF.")
            dest = tmp_path / uploaded.filename
            dest.write_bytes(await uploaded.read())

        try:
            summary = pipeline.index(tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return {"success": True, **summary}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    pipeline = get_pipeline()

    if not pipeline.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload and index PDFs first."
        )

    # Apply settings
    pipeline.config.use_reranker = req.use_reranker
    pipeline.config.top_k = req.top_k
    pipeline.config.llm_temperature = req.temperature

    try:
        result = pipeline.query(req.question, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    retrieval_method = (
        result.retrieved_chunks[0].retrieval_method if result.retrieved_chunks else "none"
    )

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence,
        faithfulness=result.faithfulness,
        hallucination_warning=result.hallucination_warning,
        latency_ms=result.latency_ms,
        retrieval_method=retrieval_method,
    )


@app.post("/api/save")
def save_index():
    try:
        get_pipeline().save()
        return {"success": True, "message": "Index saved to disk."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/load")
def load_index():
    try:
        pipeline = get_pipeline()
        pipeline.load()
        return {"success": True, "message": "Index loaded from disk.", **pipeline.stats}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/evaluation")
def evaluation():
    log_path = Path("eval_log.jsonl")
    if not log_path.exists():
        return {"records": [], "summary": {}}

    records = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    query_records = [r for r in records if r.get("event") == "query"]
    bench_records = [r for r in records if r.get("event") == "benchmark"]

    summary = {}
    if query_records:
        faith_vals = [r["faithfulness"] for r in query_records if "faithfulness" in r]
        conf_vals  = [r["confidence"]   for r in query_records if "confidence"   in r]
        lat_vals   = [r["latency_ms"]   for r in query_records if "latency_ms"   in r]

        if faith_vals:
            summary["mean_faithfulness"] = round(sum(faith_vals) / len(faith_vals), 3)
        if conf_vals:
            summary["mean_confidence"] = round(sum(conf_vals) / len(conf_vals), 3)
        if lat_vals:
            summary["mean_latency_ms"] = round(sum(lat_vals) / len(lat_vals), 1)
        summary["total_queries"] = len(query_records)

    if bench_records:
        summary["benchmark"] = bench_records[-1]

    return {
        "records": records[-50:],   # last 50 raw records
        "query_records": query_records,
        "summary": summary,
    }


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
