from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):
    """Create a FAISS vector store from document chunks and embeddings."""
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_retriever(vectorstore, k=4):
    """Return a retriever that fetches the top-k most relevant chunks."""
    return vectorstore.as_retriever(search_kwargs={"k": k})
