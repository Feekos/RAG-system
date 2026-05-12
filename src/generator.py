from __future__ import annotations

import torch
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer
from transformers import pipeline as hf_pipeline

from .config import settings

_SYSTEM_PROMPT = """\
You are a helpful multilingual assistant. Answer questions ONLY based on the provided context.
If the answer is not present in the context, reply: "The answer is not available in the provided documents."
Always respond in the SAME language the question was asked in (Russian, English, etc.).
Cite source snippets using [1], [2], etc. when referencing specific passages."""

_RAG_HUMAN_TEMPLATE = """\
Context:
{context}

Question: {question}"""


def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _RAG_HUMAN_TEMPLATE),
        ]
    )


class Generator:
    """
    Loads a Qwen causal LM as a LangChain ChatModel via HuggingFacePipeline.

    Memory requirements (approximate):
      Qwen3-4B-Instruct-2507 float16/bfloat16 = ~8 GB weights.
      CPU inference works, but is slow and defaults to float32 for compatibility.

    Device strategy:
      CUDA → device_map="auto"  (multi-GPU safe, no meta-tensor conflict)
      CPU  → device="cpu"       (no device_map to avoid meta-tensor segfault)

    Note: use `dtype` not `torch_dtype`; the latter is deprecated in recent transformers.
    """

    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.generator_model
        print(f"[Generator] Загрузка модели: {model_name} ...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        pipe_kwargs: dict = {
            "model": model_name,
            "tokenizer": tokenizer,
            "max_new_tokens": settings.max_new_tokens,
            "return_full_text": False,  # return only newly generated tokens
            "trust_remote_code": True,
            # `dtype` is the non-deprecated replacement for `torch_dtype`.
            "dtype": _resolve_dtype(getattr(settings, "torch_dtype", "auto")),
        }

        if settings.temperature > 0:
            pipe_kwargs["temperature"] = settings.temperature
            pipe_kwargs["do_sample"] = True
        else:
            pipe_kwargs["do_sample"] = False

        if torch.cuda.is_available():
            # device_map="auto" must be passed to hf_pipeline, NOT to from_pretrained,
            # to avoid calling .to() on meta-device tensors afterwards.
            pipe_kwargs["device_map"] = "auto"
        else:
            pipe_kwargs["device"] = "cpu"

        pipe = hf_pipeline("text-generation", **pipe_kwargs)
        _clear_default_max_length(pipe)
        lc_pipe = HuggingFacePipeline(pipeline=pipe)
        # ChatHuggingFace applies the tokenizer's chat template (ChatML for all Qwen variants)
        self._llm: BaseChatModel = ChatHuggingFace(llm=lc_pipe)
        print("[Generator] Готово.")

    @property
    def langchain(self) -> BaseChatModel:
        return self._llm


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


def _clear_default_max_length(pipe) -> None:
    """
    Transformers text-generation pipelines can keep max_length=20 in their
    internal forward params. When max_new_tokens is also set, generate() emits
    a noisy warning even though max_new_tokens wins.
    """
    forward_params = getattr(pipe, "_forward_params", None)
    if isinstance(forward_params, dict):
        forward_params.pop("max_length", None)

    generation_config = getattr(getattr(pipe, "model", None), "generation_config", None)
    if generation_config is not None and getattr(generation_config, "max_length", None) == 20:
        generation_config.max_length = None
