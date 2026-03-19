from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings(model_name="all-MiniLM-L6-v2"):
    """Return a HuggingFace embeddings model (runs locally, no API key needed)."""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings
