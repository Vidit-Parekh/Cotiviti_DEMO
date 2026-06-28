"""
src/ingestion/pdf.py
PDF text extraction using PyMuPDF.
Handles both digital PDFs and scanned documents.
"""

import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_path(pdf_path: str) -> str:
    """
    Extract all text from a PDF file on disk.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.
    """
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages).strip()


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract all text from PDF bytes (e.g. from a Streamlit upload).

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        Extracted text as a single string.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages).strip()


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text page by page, returning metadata per page.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        List of dicts with keys: page_number, text, char_count.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    result = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        result.append({
            "page_number": i + 1,
            "text": text,
            "char_count": len(text),
        })
    doc.close()
    return result


def is_scanned(pdf_bytes: bytes, min_chars_per_page: int = 50) -> bool:
    """
    Heuristic check: returns True if the PDF appears to be a scanned image
    (very little extractable text per page).

    Args:
        pdf_bytes: Raw PDF bytes.
        min_chars_per_page: Threshold below which a page is considered scanned.

    Returns:
        True if document appears to be mostly scanned, False otherwise.
    """
    pages = extract_pages(pdf_bytes)
    if not pages:
        return True
    avg_chars = sum(p["char_count"] for p in pages) / len(pages)
    return avg_chars < min_chars_per_page
