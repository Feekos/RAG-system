"""Tests for EmbeddingModel - LangChain HuggingFaceEmbeddings wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestEmbeddingModel:
    @pytest.fixture
    def mock_hf_embeddings(self):
        """Mock HuggingFaceEmbeddings to avoid loading a real model."""
        with patch("src.embeddings.HuggingFaceEmbeddings") as mock_cls:
            instance = MagicMock()
            instance._client.prompts = {"query": "Instruct: retrieve relevant passages\nQuery: "}
            instance._client.encode.return_value = [[0.1] * 1024, [0.2] * 1024]
            mock_cls.return_value = instance
            yield mock_cls, instance

    def test_init_creates_huggingface_embeddings(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        mock_cls, _ = mock_hf_embeddings
        EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model_name"] == "Octen/Octen-Embedding-0.6B"

    def test_init_sets_normalize_embeddings_and_left_padding(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        mock_cls, _ = mock_hf_embeddings
        EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["encode_kwargs"]["normalize_embeddings"] is True
        assert call_kwargs["model_kwargs"]["tokenizer_kwargs"]["padding_side"] == "left"

    def test_langchain_property_returns_embedding_wrapper(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")
        assert model.langchain is model

    def test_embed_documents_encodes_text_list(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")
        result = model.embed_documents(["doc 1", "doc 2"])
        mock_instance._client.encode.assert_called_once_with(
            ["doc 1", "doc 2"],
            normalize_embeddings=True,
        )
        assert len(result) == 2

    def test_embed_query_returns_list(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        mock_instance._client.encode.return_value = [[0.5] * 1024]
        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")
        result = model.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 1024

    def test_octen_query_uses_instruction_text_instead_of_prompt_name(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        mock_instance._client.encode.return_value = [[0.1] * 1024]

        model = EmbeddingModel.__new__(EmbeddingModel)
        model._client = mock_instance._client
        model._model_name = "Octen/Octen-Embedding-0.6B"
        model.embed_query("What is RAG?")

        call_text = mock_instance._client.encode.call_args.args[0][0]
        assert call_text.startswith("Instruct:")
        assert call_text.endswith("What is RAG?")
        assert "prompt_name" not in mock_instance._client.encode.call_args.kwargs

    def test_query_prompt_is_omitted_when_unavailable(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        mock_instance._client.prompts = {}
        mock_instance._client.encode.return_value = [[0.1] * 1024]

        model = EmbeddingModel.__new__(EmbeddingModel)
        model._client = mock_instance._client
        model._model_name = "intfloat/multilingual-e5-large"
        model.embed_query("What is RAG?")

        mock_instance._client.encode.assert_called_once_with(
            ["What is RAG?"],
            normalize_embeddings=True,
        )

    def test_octen_query_uses_instruction_when_prompt_unavailable(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        mock_instance._client.prompts = {}
        mock_instance._client.encode.return_value = [[0.1] * 1024]

        model = EmbeddingModel.__new__(EmbeddingModel)
        model._client = mock_instance._client
        model._model_name = "Octen/Octen-Embedding-0.6B"
        model.embed_query("What is RAG?")

        call_text = mock_instance._client.encode.call_args.args[0][0]
        assert call_text.startswith("Instruct:")
        assert call_text.endswith("What is RAG?")

    def test_embed_query_for_russian_text(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        _, mock_instance = mock_hf_embeddings
        mock_instance._client.encode.return_value = [[0.3] * 1024]
        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")
        result = model.embed_query("Что такое векторная база данных?")
        assert len(result) == 1024
        mock_instance._client.encode.assert_called_once()

    def test_embed_query_rejects_non_string_input(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")

        with pytest.raises(TypeError, match="Query text must be str"):
            model.embed_query({"text": "What is RAG?"})  # type: ignore[arg-type]

    def test_embed_documents_rejects_single_string(self, mock_hf_embeddings):
        from src.embeddings import EmbeddingModel

        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")

        with pytest.raises(TypeError, match="Document texts must be a list of str"):
            model.embed_documents("doc 1")  # type: ignore[arg-type]
