"""
diagnose_index.py — Minimal end-to-end indexing test.
Run: python diagnose_index.py
"""
import sys, traceback
sys.path.insert(0, ".")

print("=" * 60)
print("STEP 1: Import modules")
print("=" * 60)
try:
    from config import RAGConfig
    from src.loader import Document
    from src.embeddings import EmbeddingModel
    from src.chunker import SemanticChunker
    from src.hybrid_retriever import HybridRetriever
    print("[OK] All modules imported")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("STEP 2: Load config")
print("=" * 60)
try:
    cfg = RAGConfig()
    print(f"  embedding_model : {cfg.embedding_model}")
    print(f"  chunking_strategy: {cfg.chunking_strategy}")
    print(f"  use_reranker    : {cfg.use_reranker}")
    print(f"  llm_provider    : {cfg.llm_provider}")
    api_key = cfg.active_api_key()
    print(f"  api_key set     : {bool(api_key)} (len={len(api_key)})")
except Exception as e:
    print(f"[FAIL] Config: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("STEP 3: Load EmbeddingModel (will download if missing)")
print("=" * 60)
try:
    em = EmbeddingModel(
        model_name=cfg.embedding_model,
        cache_dir=cfg.embed_cache_dir,
        batch_size=cfg.embedding_batch_size,
    )
    print(f"[OK] EmbeddingModel created: {cfg.embedding_model}")
except Exception as e:
    print(f"[FAIL] EmbeddingModel: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("STEP 4: Embed sample texts (model download happens here)")
print("=" * 60)
try:
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Retrieval augmented generation improves LLM factuality.",
    ]
    vecs = em.embed_documents(sample_texts, show_progress=True)
    print(f"[OK] Embedded {len(sample_texts)} texts → shape {vecs.shape}")
except Exception as e:
    print(f"[FAIL] embed_documents: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("STEP 5: SemanticChunker")
print("=" * 60)
try:
    chunker = SemanticChunker(
        embedding_model=em,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        strategy=cfg.chunking_strategy,
        breakpoint_percentile=cfg.semantic_breakpoint_percentile,
    )
    # Create synthetic docs
    docs = [
        Document(
            content="This is page one. " * 30 + "It talks about retrieval systems.",
            metadata={"filename": "test.pdf", "page_number": 1},
        ),
        Document(
            content="Page two covers embedding models. " * 30 + "BGE is state of the art.",
            metadata={"filename": "test.pdf", "page_number": 2},
        ),
    ]
    chunks = chunker.chunk_documents(docs)
    print(f"[OK] Chunked {len(docs)} pages → {len(chunks)} chunks")
except Exception as e:
    print(f"[FAIL] SemanticChunker: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("STEP 6: Build HybridRetriever index")
print("=" * 60)
try:
    retriever = HybridRetriever(
        embedding_model=em,
        rrf_k=cfg.rrf_k,
        use_reranker=False,  # skip reranker for speed
        candidates=min(cfg.retrieval_candidates, len(chunks)),
        confidence_threshold=cfg.confidence_threshold,
    )
    retriever.build_index(chunks)
    print(f"[OK] Index built with {retriever.num_chunks} chunks")
except Exception as e:
    print(f"[FAIL] build_index: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("STEP 7: Retrieve test query")
print("=" * 60)
try:
    results = retriever.retrieve("What is BGE?", top_k=3)
    print(f"[OK] Retrieved {len(results)} chunks")
    for r in results:
        print(f"  Rank {r.rank}: score={r.score:.3f} | {r.document.content[:60]}...")
except Exception as e:
    print(f"[FAIL] retrieve: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("ALL STEPS PASSED — indexing pipeline is functional")
print("=" * 60)
