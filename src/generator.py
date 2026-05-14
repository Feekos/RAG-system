from __future__ import annotations

import logging
import os
from typing import Any

import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from pydantic import ConfigDict, PrivateAttr
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import AutoTokenizer
from transformers import pipeline as hf_pipeline
from transformers.utils import logging as hf_logging

from .config import settings

_SYSTEM_PROMPT = """\
Ты полезный многоязычный ассистент. Отвечай на вопросы только на основе предоставленного контекста.
Если ответа нет в контексте, ответь: "Ответ отсутствует в предоставленных документах."
Всегда отвечай на том же языке, на котором задан вопрос.
Не выводи ход рассуждений, Thinking Process, analysis, hidden reasoning, планы или промежуточные шаги.
Выводи только финальный ответ для пользователя.
Когда ссылаешься на конкретные фрагменты, указывай источники в формате [1], [2] и так далее.
Отвечай кратко и законченными предложениями. Не обрывай ответ на середине предложения."""

_NO_THINKING_PROMPT = (
    "Не выводи ход рассуждений, Thinking Process, analysis, hidden reasoning, "
    "планы или промежуточные шаги. Выводи только финальный ответ для пользователя."
)

_RAG_HUMAN_TEMPLATE = """\
Conversation history:
{chat_history}

Context:
{context}

Answer language:
{answer_language}

Question: {question}

/no_think
Answer directly. Do not include a thinking process."""


def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _get_system_prompt()),
            ("human", _RAG_HUMAN_TEMPLATE),
        ]
    ).partial(chat_history="No prior conversation.")


def _get_system_prompt() -> str:
    configured_prompt = getattr(settings, "system_prompt", "")
    if isinstance(configured_prompt, str) and configured_prompt.strip():
        return f"{configured_prompt.strip()}\n{_NO_THINKING_PROMPT}"
    return _SYSTEM_PROMPT


class Generator:
    """
    Loads a Qwen model as a LangChain ChatModel.

    Memory requirements (approximate):
      Qwen3.5-4B float16/bfloat16 = ~8 GB weights plus vision encoder overhead.
      CPU inference works, but is slow and defaults to float32 for compatibility.

    Device strategy:
      CUDA → device_map="auto"  (multi-GPU safe, no meta-tensor conflict)
      CPU  → device="cpu"       (no device_map to avoid meta-tensor segfault)

    Note: use `dtype` not `torch_dtype`; the latter is deprecated in recent transformers.
    """

    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.generator_model
        _configure_transformers_logging()
        print(f"[Generator] Загрузка модели: {model_name} ...")

        if _uses_image_text_to_text_model(model_name):
            self._llm = QwenImageTextChatModel(
                model_name=model_name,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                dtype=_resolve_dtype(getattr(settings, "torch_dtype", "auto")),
            )
            print("[Generator] Готово.")
            return

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.clean_up_tokenization_spaces = False

        pipe_kwargs: dict = {
            "model": model_name,
            "tokenizer": tokenizer,
            "max_new_tokens": settings.max_new_tokens,
            "clean_up_tokenization_spaces": False,
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
        _normalize_generation_config(pipe)
        lc_pipe = HuggingFacePipeline(pipeline=pipe)
        # ChatHuggingFace applies the tokenizer's chat template (ChatML for all Qwen variants)
        self._llm: BaseChatModel = ChatHuggingFace(llm=lc_pipe)
        print("[Generator] Готово.")

    @property
    def langchain(self) -> BaseChatModel:
        return self._llm


class QwenImageTextChatModel(BaseChatModel):
    """Minimal LangChain chat wrapper for Qwen3.5 image-text-to-text checkpoints."""

    model_name: str
    max_new_tokens: int
    temperature: float

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _processor: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, *, model_name: str, max_new_tokens: int, temperature: float, dtype: Any):
        super().__init__(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self._processor = _load_qwen35_processor(model_name)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dtype,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        self._model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "qwen-image-text-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        hf_messages = [_to_qwen_message(message) for message in messages]
        inputs = _apply_qwen_chat_template(
            self._processor,
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **generate_kwargs)

        prompt_tokens = inputs["input_ids"].shape[-1]
        text = self._processor.decode(
            outputs[0][prompt_tokens:],
            skip_special_tokens=True,
        ).strip()
        text = _clean_model_answer(text)
        if stop:
            text = _truncate_at_stop(text, stop)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _uses_image_text_to_text_model(model_name: str) -> bool:
    return "qwen3.5" in model_name.lower()


def _load_qwen35_processor(model_name: str):
    try:
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except ImportError as exc:
        if "Torchvision" not in str(exc) and "torchvision" not in str(exc):
            raise
        print(
            "[Generator] torchvision не найден; используется text-only tokenizer для Qwen3.5. "
            "Для image/video inputs установите torchvision и пересоберите Docker-образ."
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.clean_up_tokenization_spaces = False
        return tokenizer


def _to_qwen_message(message: BaseMessage) -> dict[str, Any]:
    role_by_type = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }
    role = role_by_type.get(message.type, message.type)
    return {"role": role, "content": _to_qwen_content(message.content)}


def _to_qwen_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def _strip_qwen_thinking(text: str) -> str:
    marker = "</think>"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text


def _apply_qwen_chat_template(processor, messages: list[dict[str, Any]], **kwargs: Any):
    try:
        return processor.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return processor.apply_chat_template(messages, **kwargs)


def _clean_model_answer(text: str) -> str:
    text = _strip_qwen_thinking(str(text)).strip()
    for marker in ("Final Answer:", "Final answer:", "Ответ:", "Итоговый ответ:"):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    if text.lower().startswith("thinking process:"):
        parts = text.split("\n\n", 1)
        text = parts[1].strip() if len(parts) > 1 else ""
    return text


def _truncate_at_stop(text: str, stop: list[str]) -> str:
    cut = len(text)
    for token in stop:
        index = text.find(token)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].rstrip()


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


def _configure_transformers_logging() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    for logger_name in (
        "transformers.generation.configuration_utils",
        "transformers.generation.utils",
        "transformers.pipelines.base",
        "transformers.tokenization_utils_base",
    ):
        hf_logging.get_logger(logger_name).setLevel(logging.ERROR)


def _normalize_generation_config(pipe) -> None:
    """
    Keep generation settings explicit so transformers does not print first-call
    generation_config notices before the assistant's answer.
    """
    forward_params = getattr(pipe, "_forward_params", None)
    if isinstance(forward_params, dict):
        forward_params.pop("max_length", None)
        forward_params["max_new_tokens"] = settings.max_new_tokens
        forward_params["do_sample"] = settings.temperature > 0
        if settings.temperature > 0:
            forward_params["temperature"] = settings.temperature
        else:
            forward_params.pop("temperature", None)

    generation_config = getattr(getattr(pipe, "model", None), "generation_config", None)
    if generation_config is not None:
        generation_config.max_new_tokens = settings.max_new_tokens
        generation_config.do_sample = settings.temperature > 0
        generation_config.max_length = None
        if settings.temperature > 0:
            generation_config.temperature = settings.temperature
        setattr(generation_config, "_from_model_config", False)
