from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from .config import settings


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingModel(Embeddings):
    """LangChain embeddings wrapper for Octen/Qwen-style text embeddings."""

    def __init__(self, model_name: str | None = None):
        name = model_name or settings.embedding_model
        self._device = _auto_device()
        print(f"[Embeddings] Загрузка модели: {name} на {self._device} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            padding_side="left",
            use_fast=False,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModel.from_pretrained(
            name,
            trust_remote_code=True,
            dtype=_resolve_dtype(getattr(settings, "torch_dtype", "auto")),
        ).to(self._device)
        self._model.eval()
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
        return self._encode_texts([self._format_query_text(text)])[0]

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        batch = self._tokenize_texts(texts)
        batch = {key: value.to(self._device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = self._model(**batch)
            embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().float().tolist()

    def _tokenize_texts(self, texts: List[str]) -> dict:
        encoded = [
            self._tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=8192,
                return_attention_mask=True,
            )
            for text in texts
        ]
        return self._tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
        )

    def _format_query_text(self, text: str) -> str:
        model_name = getattr(self, "_model_name", settings.embedding_model).lower()
        if "octen-embedding" in model_name or "qwen3-embedding" in model_name:
            return (
                "Instruct: Given a web search query, retrieve relevant passages "
                f"that answer the query\nQuery: {text}"
            )
        return text


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _resolve_dtype(dtype_name: str):
    if not isinstance(dtype_name, str):
        dtype_name = "auto"
    normalized = dtype_name.lower().strip()
    if normalized == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(
        "Unsupported TORCH_DTYPE value. Use one of: auto, float16, bfloat16, float32."
    )
