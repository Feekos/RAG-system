from __future__ import annotations

import time
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import settings
from .embeddings import EmbeddingModel


class VectorStore:
    """Wraps LangChain QdrantVectorStore with explicit collection lifecycle management."""

    def __init__(self, embeddings: EmbeddingModel):
        self._client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self._ensure_collection()
        self._lc = QdrantVectorStore(
            client=self._client,
            collection_name=settings.qdrant_collection,
            embedding=embeddings.langchain,
        )

    def _ensure_collection(self) -> None:
        existing = self._get_collection_names()
        if settings.qdrant_collection not in existing:
            self._client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[VectorStore] Created '{settings.qdrant_collection}' (dim={settings.embedding_dim})")
        else:
            print(f"[VectorStore] Found existing collection '{settings.qdrant_collection}'")

    def _get_collection_names(self) -> set[str]:
        last_error: Exception | None = None
        for attempt in range(1, 31):
            try:
                return {c.name for c in self._client.get_collections().collections}
            except Exception as exc:
                last_error = exc
                if attempt == 30:
                    break
                time.sleep(1)
        raise RuntimeError(
            f"Cannot connect to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}"
        ) from last_error

    def add_documents(self, docs: List[Document]) -> List[str]:
        return self._lc.add_documents(docs)

    def as_retriever(self, top_k: int | None = None) -> VectorStoreRetriever:
        return self._lc.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k or settings.top_k},
        )

    def count(self) -> int:
        return self._client.count(settings.qdrant_collection).count

    def delete_collection(self) -> None:
        self._client.delete_collection(settings.qdrant_collection)
        print(f"[VectorStore] Deleted '{settings.qdrant_collection}'")
