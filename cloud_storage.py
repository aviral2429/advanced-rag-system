"""
cloud_storage.py — Supabase Storage integration for PDF RAG Chatbot

Uses service role key (preferred) for server-side operations, 
falling back to anon key. Fixes RLS policy 403 errors.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()  # Load .env file automatically

BUCKET = "pdfs"
_client: Client | None = None


def _get_client() -> Client:
    """Return a Supabase client using service role key (preferred) or anon key."""
    global _client
    if _client is not None:
        return _client

    url = os.getenv("SUPABASE_URL")
    # Prefer service role key (bypasses RLS), fall back to anon key
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        raise ValueError(
            "Supabase credentials not set. Add to your .env:\n"
            "  SUPABASE_URL            = https://your-project.supabase.co\n"
            "  SUPABASE_SERVICE_ROLE_KEY = your-service-role-key  (preferred)\n"
            "  SUPABASE_ANON_KEY         = your-anon-key           (fallback)"
        )

    _client = create_client(url, key)
    return _client


# ── Public API ────────────────────────────────────────────────────────────────

def upload_pdf(file_bytes: bytes, filename: str) -> str:
    """Upload PDF bytes to Supabase Storage, overwriting if exists."""
    client = _get_client()
    # Remove first to avoid 409 duplicate conflict
    try:
        client.storage.from_(BUCKET).remove([filename])
    except Exception:
        pass
    client.storage.from_(BUCKET).upload(
        path=filename,
        file=file_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )
    return filename


def list_pdfs() -> list[str]:
    """List all PDF filenames in the bucket."""
    client = _get_client()
    response = client.storage.from_(BUCKET).list()
    return [
        item["name"]
        for item in response
        if isinstance(item, dict) and item.get("name", "").endswith(".pdf")
    ]


def get_pdf_download_url(filename: str) -> str:
    """Get a public download URL for a stored PDF."""
    client = _get_client()
    return client.storage.from_(BUCKET).get_public_url(filename)


def delete_pdf(filename: str) -> bool:
    """Delete a PDF from Supabase Storage."""
    client = _get_client()
    client.storage.from_(BUCKET).remove([filename])
    return True
