"""Tests for EmbeddingModel — LangChain HuggingFaceEmbeddings wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestEmbeddingModel:
    @pytest.fixture
    def mock_hf_embeddings(self):
        """Mock HuggingFaceEmbeddings to avoid loading a real model."""
        with patch("src.embeddings.HuggingFaceEmbeddings") as mock_cls:
            instance = MagicMock()
            instance.embed_query.return_value = [0.1] * 1024
            instance.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]
            mock_cls.return_value = instance
            yield mock_cls, instance

    def test_init_creates_huggingface_embeddings(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        mock_cls, _ = mock_hf_embeddings
        EmbeddingModel(model_name="BAAI/bge-m3")
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model_name"] == "BAAI/bge-m3"

    def test_init_sets_normalize_embeddings(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        mock_cls, _ = mock_hf_embeddings
        EmbeddingModel(model_name="BAAI/bge-m3")
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["encode_kwargs"]["normalize_embeddings"] is True

    def test_langchain_property_returns_embedding_wrapper(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        model = EmbeddingModel(model_name="BAAI/bge-m3")
        assert model.langchain is model

    def test_embed_documents_delegates_to_huggingface_embeddings(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        model = EmbeddingModel(model_name="BAAI/bge-m3")
        result = model.embed_documents(["doc 1", "doc 2"])
        mock_instance.embed_documents.assert_called_once_with(["doc 1", "doc 2"])
        assert len(result) == 2

    def test_embed_query_returns_list(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        mock_instance.embed_query.return_value = [0.5] * 1024
        model = EmbeddingModel(model_name="BAAI/bge-m3")
        result = model.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 1024

    def test_bge_model_adds_query_prefix(self, mock_hf_embeddings):
        """BGE models require a specific prefix for asymmetric retrieval queries."""
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        mock_instance.embed_query.return_value = [0.1] * 1024

        with patch("src.embeddings.settings") as mock_settings:
            mock_settings.embedding_model = "BAAI/bge-m3"
            model = EmbeddingModel.__new__(EmbeddingModel)
            model._lc = mock_instance
            model.embed_query("Что такое RAG?")

        call_args = mock_instance.embed_query.call_args[0][0]
        assert "Represent this sentence" in call_args
        assert "Что такое RAG?" in call_args

    def test_non_bge_model_does_not_add_prefix(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        mock_instance.embed_query.return_value = [0.1] * 1024

        with patch("src.embeddings.settings") as mock_settings:
            mock_settings.embedding_model = "intfloat/multilingual-e5-large"
            model = EmbeddingModel.__new__(EmbeddingModel)
            model._lc = mock_instance
            model.embed_query("What is RAG?")

        call_args = mock_instance.embed_query.call_args[0][0]
        assert call_args == "What is RAG?"

    def test_embed_query_for_russian_text(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        mock_instance.embed_query.return_value = [0.3] * 1024
        model = EmbeddingModel(model_name="BAAI/bge-m3")
        result = model.embed_query("Что такое векторная база данных?")
        assert len(result) == 1024
        mock_instance.embed_query.assert_called_once()
