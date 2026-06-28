"""
src/rag/vectorstore.py
In-memory vector store for RAG retrieval.
Wraps the TF-IDF index from embeddings.py and exposes a simple
retrieve() interface used by the Q&A assistant.
No external dependencies or API calls required.
"""

from src.rag.embeddings import (
    chunk_text,
    build_tfidf_index,
    embed_query,
    cosine_similarity,
)


class InMemoryVectorStore:
    """
    Lightweight in-memory vector store for clinical document RAG.

    Usage:
        store = InMemoryVectorStore()
        store.index_document(document_text)
        results = store.retrieve("what medications were prescribed", top_k=3)
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._index: dict | None = None
        self._document_text: str = ""

    def index_document(self, text: str) -> int:
        """
        Chunk and index a clinical document.

        Args:
            text: Full document text.

        Returns:
            Number of chunks indexed.
        """
        self._document_text = text
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        self._index = build_tfidf_index(chunks)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language question or search string.
            top_k: Number of top chunks to return.

        Returns:
            List of chunk dicts sorted by relevance, each with an added
            'score' key containing the cosine similarity.

        Raises:
            RuntimeError: If index_document() has not been called yet.
        """
        if self._index is None:
            raise RuntimeError("No document indexed. Call index_document() first.")

        q_vec = embed_query(query, self._index["idf"])
        scored = []
        for chunk, vec in zip(self._index["chunks"], self._index["vectors"]):
            score = cosine_similarity(q_vec, vec)
            scored.append({**chunk, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def retrieve_context(self, query: str, top_k: int = 3, max_chars: int = 2000) -> str:
        """
        Retrieve and format relevant context as a string for LLM prompting.

        Args:
            query: Natural language question.
            top_k: Number of top chunks to retrieve.
            max_chars: Maximum total characters of context to return.

        Returns:
            Formatted context string ready to inject into an LLM prompt.
        """
        chunks = self.retrieve(query, top_k=top_k)
        parts = []
        total = 0
        for chunk in chunks:
            text = chunk["text"]
            if total + len(text) > max_chars:
                text = text[: max_chars - total]
            parts.append(f"[Chunk {chunk['chunk_id'] + 1}]\n{text}")
            total += len(text)
            if total >= max_chars:
                break
        return "\n\n".join(parts)

    @property
    def is_ready(self) -> bool:
        """True if a document has been indexed."""
        return self._index is not None

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the current index."""
        if self._index is None:
            return 0
        return len(self._index["chunks"])
