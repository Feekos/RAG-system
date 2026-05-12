from __future__ import annotations

from typing import Iterable, List

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingModel(Embeddings):
    """LangChain embeddings wrapper for multilingual sentence-transformer models."""

    def __init__(self, model_name: str | None = None):
        name = model_name or settings.embedding_model
        print(f"[Embeddings] Загрузка модели: {name} на {_auto_device()} ...")
        self._lc = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={
                "device": _auto_device(),
                "tokenizer_kwargs": {"padding_side": "left"},
            },
            encode_kwargs={"normalize_embeddings": True},
        )
        self._client = self._lc._client
        self._model_name = name

    @property
    def langchain(self) -> Embeddings:
        return self

    @staticmethod
    def _ensure_text(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be str, got {type(value).__name__}")
        return value

    @classmethod
    def _ensure_texts(cls, values: Iterable[object]) -> List[str]:
        if isinstance(values, str):
            raise TypeError("Document texts must be a list of str, got str")
        return [cls._ensure_text(value, "Document text") for value in values]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode_texts(self._ensure_texts(texts))

    def embed_query(self, text: str) -> List[float]:
        text = self._ensure_text(text, "Query text")
        prompt_name = self._query_prompt_name()
        if prompt_name is not None:
            return self._encode_texts([text], prompt_name=prompt_name)[0]
        return self._encode_texts([self._format_query_text(text)])[0]

    def _encode_texts(self, texts: List[str], **encode_kwargs) -> List[List[float]]:
        embeddings = self._client.encode(
            texts,
            normalize_embeddings=True,
            **encode_kwargs,
        )
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return [
            embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            for embedding in embeddings
        ]

    def _query_prompt_name(self) -> str | None:
        prompts = getattr(self._client, "prompts", None)
        if isinstance(prompts, dict) and "query" in prompts:
            return "query"
        return None

    def _format_query_text(self, text: str) -> str:
        model_name = getattr(self, "_model_name", settings.embedding_model).lower()
        if "octen-embedding" in model_name or "qwen3-embedding" in model_name:
            return (
                "Instruct: Given a web search query, retrieve relevant passages "
                f"that answer the query\nQuery: {text}"
            )
        return text
