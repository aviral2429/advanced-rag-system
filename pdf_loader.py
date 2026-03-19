from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
import requests


def load_pdf(uploaded_file):
    """Load a PDF from a Streamlit uploaded file object and return list of Documents."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    return documents


def load_pdf_from_url(url: str) -> list:
    """
    Download a PDF from a URL (e.g. Firebase signed URL) and load it as LangChain Documents.

    Args:
        url: HTTPS URL pointing to the PDF file.

    Returns:
        List of LangChain Document objects (one per page).
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    return documents
