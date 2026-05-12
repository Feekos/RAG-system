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
            model_kwargs={"device": _auto_device()},
            encode_kwargs={"normalize_embeddings": True},
        )
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
        return self._lc.embed_documents(self._ensure_texts(texts))

    def embed_query(self, text: str) -> List[float]:
        text = self._ensure_text(text, "Query text")
        # BGE family uses asymmetric retrieval; queries benefit from this instruction.
        model_name = getattr(self, "_model_name", settings.embedding_model)
        if "bge" in model_name.lower():
            text = f"Represent this sentence for searching relevant passages: {text}"
        return self._lc.embed_query(text)
