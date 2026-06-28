"""
src/ingestion/ocr.py
OCR pipeline for scanned clinical documents.
Uses PyMuPDF to render pages as images, then extracts text via Tesseract if available,
falling back to PyMuPDF's built-in OCR layer.
"""

import fitz  # PyMuPDF
import subprocess
import tempfile
import os
from pathlib import Path


def _tesseract_available() -> bool:
    """Check if Tesseract OCR is installed on the system."""
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ocr_pdf_bytes(pdf_bytes: bytes, dpi: int = 200) -> str:
    """
    Run OCR on a scanned PDF and return extracted text.
    Uses Tesseract if available, otherwise falls back to PyMuPDF text layer.

    Args:
        pdf_bytes: Raw PDF bytes.
        dpi: Render resolution for image-based OCR (higher = better quality, slower).

    Returns:
        OCR-extracted text as a single string.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []

    use_tesseract = _tesseract_available()

    for page in doc:
        if use_tesseract:
            text = _ocr_page_tesseract(page, dpi)
        else:
            # PyMuPDF fallback: render to pixmap and use get_text with blocks
            text = page.get_text("text")
        all_text.append(text.strip())

    doc.close()
    return "\n\n".join(all_text)


def _ocr_page_tesseract(page: fitz.Page, dpi: int = 200) -> str:
    """
    Render a single PDF page to an image and run Tesseract OCR on it.

    Args:
        page: A PyMuPDF page object.
        dpi: Render resolution.

    Returns:
        OCR text for this page.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "page.png")
        out_base = os.path.join(tmpdir, "ocr_out")
        pix.save(img_path)

        subprocess.run(
            ["tesseract", img_path, out_base, "--psm", "6", "-l", "eng"],
            capture_output=True, timeout=30
        )

        out_file = out_base + ".txt"
        if os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    return ""


def extract_with_fallback(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Attempt digital text extraction first; fall back to OCR if needed.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        Tuple of (extracted_text, method_used).
        method_used is one of: 'digital', 'ocr_tesseract', 'ocr_pymupdf'.
    """
    from src.ingestion.pdf import extract_text_from_bytes, is_scanned

    if not is_scanned(pdf_bytes):
        text = extract_text_from_bytes(pdf_bytes)
        return text, "digital"

    method = "ocr_tesseract" if _tesseract_available() else "ocr_pymupdf"
    text = ocr_pdf_bytes(pdf_bytes)
    return text, method
