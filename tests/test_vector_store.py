"""Tests for VectorStore — Qdrant collection management and document operations."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from langchain_core.documents import Document


def _make_collection(name: str):
    c = MagicMock()
    c.name = name
    return c


class TestEnsureCollection:
    def test_creates_collection_when_missing(self):
        with (
            patch("src.vector_store.QdrantClient") as mock_client_cls,
            patch("src.vector_store.QdrantVectorStore"),
        ):
            mock_client = MagicMock()
            mock_client.get_collections.return_value.collections = []
            mock_client_cls.return_value = mock_client

            from src.embeddings import EmbeddingModel
            mock_embeddings = MagicMock(spec=EmbeddingModel)
            mock_embeddings.langchain = MagicMock()

            from src.vector_store import VectorStore
            VectorStore(mock_embeddings)

            mock_client.create_collection.assert_called_once()

    def test_skips_creation_when_collection_exists(self):
        with (
            patch("src.vector_store.QdrantClient") as mock_client_cls,
            patch("src.vector_store.QdrantVectorStore"),
            patch("src.vector_store.settings") as mock_settings,
        ):
            mock_settings.qdrant_host = "localhost"
            mock_settings.qdrant_port = 6333
            mock_settings.qdrant_collection = "documents"
            mock_settings.embedding_dim = 2560
            mock_settings.top_k = 5

            mock_client = MagicMock()
            mock_client.get_collections.return_value.collections = [
                _make_collection("documents")
            ]
            mock_client_cls.return_value = mock_client

            from src.embeddings import EmbeddingModel
            mock_embeddings = MagicMock(spec=EmbeddingModel)
            mock_embeddings.langchain = MagicMock()

            from src.vector_store import VectorStore
            VectorStore(mock_embeddings)

            mock_client.create_collection.assert_not_called()

    def test_collection_uses_cosine_distance(self):
        with (
            patch("src.vector_store.QdrantClient") as mock_client_cls,
            patch("src.vector_store.QdrantVectorStore"),
            patch("src.vector_store.settings") as mock_settings,
        ):
            mock_settings.qdrant_collection = "documents"
            mock_settings.qdrant_host = "localhost"
            mock_settings.qdrant_port = 6333
            mock_settings.embedding_dim = 2560
            mock_settings.top_k = 5

            mock_client = MagicMock()
            mock_client.get_collections.return_value.collections = []
            mock_client_cls.return_value = mock_client

            mock_embeddings = MagicMock()
            mock_embeddings.langchain = MagicMock()

            from src.vector_store import VectorStore
            from qdrant_client.models import Distance
            VectorStore(mock_embeddings)

            call_kwargs = mock_client.create_collection.call_args.kwargs
            assert call_kwargs["vectors_config"].distance == Distance.COSINE

    def test_existing_collection_dimension_mismatch_raises(self):
        with (
            patch("src.vector_store.QdrantClient") as mock_client_cls,
            patch("src.vector_store.QdrantVectorStore"),
            patch("src.vector_store.settings") as mock_settings,
        ):
            mock_settings.qdrant_collection = "documents"
            mock_settings.qdrant_host = "localhost"
            mock_settings.qdrant_port = 6333
            mock_settings.embedding_dim = 2560
            mock_settings.top_k = 5

            collection_info = MagicMock()
            collection_info.config.params.vectors.size = 1024

            mock_client = MagicMock()
            mock_client.get_collections.return_value.collections = [_make_collection("documents")]
            mock_client.get_collection.return_value = collection_info
            mock_client_cls.return_value = mock_client

            mock_embeddings = MagicMock()
            mock_embeddings.langchain = MagicMock()

            from src.vector_store import VectorStore

            with pytest.raises(RuntimeError, match="vector size 1024"):
                VectorStore(mock_embeddings)


class TestAddDocuments:
    @pytest.fixture
    def store_with_mocks(self):
        with (
            patch("src.vector_store.QdrantClient") as mock_client_cls,
            patch("src.vector_store.QdrantVectorStore") as mock_qdrant_cls,
            patch("src.vector_store.settings") as mock_settings,
        ):
            mock_settings.qdrant_collection = "documents"
            mock_settings.qdrant_host = "localhost"
            mock_settings.qdrant_port = 6333
            mock_settings.embedding_dim = 2560
            mock_settings.top_k = 5

            mock_client = MagicMock()
            mock_client.get_collections.return_value.collections = [_make_collection("documents")]
            mock_client_cls.return_value = mock_client

            mock_lc = MagicMock()
            mock_lc.add_documents.return_value = ["id1", "id2"]
            mock_qdrant_cls.return_value = mock_lc

            mock_embeddings = MagicMock()
            mock_embeddings.langchain = MagicMock()

            from src.vector_store import VectorStore
            store = VectorStore(mock_embeddings)
            yield store, mock_lc, mock_client

    def test_add_documents_delegates_to_lc_store(self, store_with_mocks):
        store, mock_lc, _ = store_with_mocks
        docs = [Document(page_content="Test", metadata={"source": "test.txt"})]
        ids = store.add_documents(docs)
        mock_lc.add_documents.assert_called_once_with(docs)
        assert ids == ["id1", "id2"]

    def test_count_delegates_to_qdrant_client(self, store_with_mocks):
        store, _, mock_client = store_with_mocks
        mock_client.count.return_value.count = 42
        assert store.count() == 42

    def test_delete_collection_calls_client(self, store_with_mocks):
        store, _, mock_client = store_with_mocks
        store.delete_collection()
        mock_client.delete_collection.assert_called_once()

    def test_as_retriever_returns_retriever(self, store_with_mocks):
        store, mock_lc, _ = store_with_mocks
        mock_lc.as_retriever.return_value = MagicMock()
        retriever = store.as_retriever(top_k=3)
        mock_lc.as_retriever.assert_called_once()
        call_kwargs = mock_lc.as_retriever.call_args.kwargs
        assert call_kwargs["search_kwargs"]["k"] == 3


class TestConnectionParameters:
    def test_connects_to_configured_host_and_port(self):
        with (
            patch("src.vector_store.QdrantClient") as mock_client_cls,
            patch("src.vector_store.QdrantVectorStore"),
            patch("src.vector_store.settings") as mock_settings,
        ):
            mock_settings.qdrant_host = "qdrant-server"
            mock_settings.qdrant_port = 6333
            mock_settings.qdrant_collection = "my_collection"
            mock_settings.embedding_dim = 2560
            mock_settings.top_k = 5

            mock_client = MagicMock()
            mock_client.get_collections.return_value.collections = []
            mock_client_cls.return_value = mock_client

            mock_embeddings = MagicMock()
            mock_embeddings.langchain = MagicMock()

            from src.vector_store import VectorStore
            VectorStore(mock_embeddings)

            mock_client_cls.assert_called_once_with(host="qdrant-server", port=6333)
