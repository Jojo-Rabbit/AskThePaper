"""
VectorStore — wraps ChromaDB.
Handles PDF loading, recursive text splitting, embedding, and similarity search.
"""

import os
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"   # fast, local, no API key needed
        )
        self._collection = self._client.get_or_create_collection(
            name="research_docs",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str, doc_id: str, source: str) -> int:
        """Load, split, and embed a PDF. Returns number of chunks indexed."""
        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()                          # list of Document objects
        chunks = self._splitter.split_documents(pages)

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            ids.append(chunk_id)
            docs.append(chunk.page_content)
            metas.append({
                "source": source,
                "doc_id": doc_id,
                "page":   chunk.metadata.get("page", 0),
                "chunk":  i,
            })

        # Upsert — safe to re-ingest the same PDF
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = Config.TOP_K_RETRIEVAL) -> list[dict[str, Any]]:
        """
        Return top_k most relevant chunks for query.
        Each result: { text, source, page, score }
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text":   text,
                "source": meta.get("source", "unknown"),
                "page":   meta.get("page", 0) + 1,   # 1-indexed for display
                "score":  round(1 - dist, 3),         # cosine similarity
            })
        return hits

    # ── Housekeeping ───────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[str]:
        if self._collection.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in all_meta})
