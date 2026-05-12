"""Tests for EmbeddingModel direct AutoTokenizer/AutoModel wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestEmbeddingModel:
    @pytest.fixture
    def mock_transformers(self):
        with (
            patch("src.embeddings.AutoTokenizer") as mock_tokenizer_cls,
            patch("src.embeddings.AutoModel") as mock_model_cls,
            patch("src.embeddings.torch") as mock_torch,
            patch("src.embeddings.F") as mock_f,
        ):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.float32 = "float32"
            mock_torch.float16 = "float16"
            mock_torch.no_grad.return_value.__enter__.return_value = None
            mock_torch.no_grad.return_value.__exit__.return_value = None

            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "<|endoftext|>"
            mock_tokenizer.eos_token = "<|endoftext|>"
            mock_tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}
            mock_tokenizer.pad.return_value = {
                "input_ids": MagicMock(),
                "attention_mask": MagicMock(),
            }
            for value in mock_tokenizer.pad.return_value.values():
                value.to.return_value = value
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model_cls.from_pretrained.return_value = mock_model

            mock_embedding = MagicMock()
            mock_embedding.detach.return_value.cpu.return_value.float.return_value.tolist.return_value = [
                [0.1] * 1024,
                [0.2] * 1024,
            ]
            mock_f.normalize.return_value = mock_embedding

            yield {
                "tokenizer_cls": mock_tokenizer_cls,
                "tokenizer": mock_tokenizer,
                "model_cls": mock_model_cls,
                "model": mock_model,
                "torch": mock_torch,
                "normalize": mock_f.normalize,
            }

    def test_init_loads_tokenizer_and_model(self, mock_transformers):
        from src.embeddings import EmbeddingModel

        EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")

        mock_transformers["tokenizer_cls"].from_pretrained.assert_called_once_with(
            "Octen/Octen-Embedding-0.6B",
            trust_remote_code=True,
            padding_side="left",
            use_fast=False,
        )
        mock_transformers["model_cls"].from_pretrained.assert_called_once()

    def test_langchain_property_returns_embedding_wrapper(self, mock_transformers):
        from src.embeddings import EmbeddingModel

        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")
        assert model.langchain is model

    def test_embed_documents_tokenizes_text_list(self, mock_transformers):
        from src.embeddings import EmbeddingModel

        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")
        result = model.embed_documents(["doc 1", "doc 2"])

        assert mock_transformers["tokenizer"].call_count == 2
        assert mock_transformers["tokenizer"].call_args_list[0].args[0] == "doc 1"
        assert mock_transformers["tokenizer"].call_args_list[1].args[0] == "doc 2"
        call_kwargs = mock_transformers["tokenizer"].call_args_list[0].kwargs
        assert call_kwargs["padding"] is False
        assert call_kwargs["truncation"] is True
        mock_transformers["tokenizer"].pad.assert_called_once()
        assert mock_transformers["tokenizer"].pad.call_args.kwargs["return_tensors"] == "pt"
        assert len(result) == 2

    def test_embed_query_returns_list(self, mock_transformers):
        from src.embeddings import EmbeddingModel

        mock_transformers[
            "normalize"
        ].return_value.detach.return_value.cpu.return_value.float.return_value.tolist.return_value = [[0.5] * 1024]
        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")

        result = model.embed_query("test query")

        assert isinstance(result, list)
        assert len(result) == 1024

    def test_octen_query_uses_instruction_text(self, mock_transformers):
        from src.embeddings import EmbeddingModel

        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")
        model.embed_query("What is RAG?")

        call_text = mock_transformers["tokenizer"].call_args.args[0]
        assert call_text.startswith("Instruct:")
        assert call_text.endswith("What is RAG?")

    def test_query_instruction_is_omitted_for_other_models(self, mock_transformers):
        from src.embeddings import EmbeddingModel

        model = EmbeddingModel(model_name="intfloat/multilingual-e5-large")
        model.embed_query("What is RAG?")

        assert mock_transformers["tokenizer"].call_args.args[0] == "What is RAG?"

    def test_embed_query_rejects_non_string_input(self, mock_transformers):
        from src.embeddings import EmbeddingModel

        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")

        with pytest.raises(TypeError, match="Query text must be str"):
            model.embed_query({"text": "What is RAG?"})  # type: ignore[arg-type]

    def test_embed_documents_rejects_single_string(self, mock_transformers):
        from src.embeddings import EmbeddingModel

        model = EmbeddingModel(model_name="Octen/Octen-Embedding-0.6B")

        with pytest.raises(TypeError, match="Document texts must be a list of str"):
            model.embed_documents("doc 1")  # type: ignore[arg-type]
