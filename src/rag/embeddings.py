"""
src/rag/embeddings.py
Lightweight document embedding pipeline for RAG.
Chunks the clinical document into overlapping windows and produces
TF-IDF style sparse embeddings for fast in-memory retrieval.
No external embedding API required - fully local and free.
"""

import re
import math
from collections import Counter


# -------------------------------------------------------
# Text chunking
# -------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[dict]:
    """
    Split text into overlapping chunks for retrieval.

    Args:
        text: Full document text.
        chunk_size: Approximate characters per chunk.
        overlap: Overlap between consecutive chunks (characters).

    Returns:
        List of dicts with keys: chunk_id, text, start, end.
    """
    text = text.strip()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at a sentence boundary
        if end < len(text):
            boundary = text.rfind(".", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "start": start,
                "end": end,
            })
            chunk_id += 1

        start = end - overlap if end < len(text) else len(text)

    return chunks


# -------------------------------------------------------
# TF-IDF sparse embeddings
# -------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def _tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency for a list of tokens."""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: count / total for term, count in counts.items()}


def _idf(chunks: list[dict]) -> dict[str, float]:
    """Compute inverse document frequency across all chunks."""
    n = len(chunks)
    df: dict[str, int] = {}
    for chunk in chunks:
        tokens = set(_tokenize(chunk["text"]))
        for token in tokens:
            df[token] = df.get(token, 0) + 1
    return {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}


def build_tfidf_index(chunks: list[dict]) -> dict:
    """
    Build a TF-IDF index over document chunks.

    Args:
        chunks: List of chunk dicts from chunk_text().

    Returns:
        Index dict with keys: chunks, idf, vectors.
    """
    idf = _idf(chunks)
    vectors = []
    for chunk in chunks:
        tokens = _tokenize(chunk["text"])
        tf = _tf(tokens)
        vec = {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}
        vectors.append(vec)
    return {"chunks": chunks, "idf": idf, "vectors": vectors}


def embed_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    """
    Embed a query string using the same IDF weights as the index.

    Args:
        query: Search query.
        idf: IDF weights from build_tfidf_index().

    Returns:
        Sparse TF-IDF vector for the query.
    """
    tokens = _tokenize(query)
    tf = _tf(tokens)
    return {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two sparse vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
