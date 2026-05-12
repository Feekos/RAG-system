"""Tests for Generator — prompt structure, LangChain wiring, and multilingual system prompt."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ================================================================== build_rag_prompt (pure function)

class TestBuildRagPrompt:
    @pytest.fixture(autouse=True)
    def import_prompt_builder(self):
        from src.generator import build_rag_prompt, _SYSTEM_PROMPT, _RAG_HUMAN_TEMPLATE, _get_system_prompt
        self.build = build_rag_prompt
        self.system = _SYSTEM_PROMPT
        self.human_template = _RAG_HUMAN_TEMPLATE
        self.get_system_prompt = _get_system_prompt

    def test_prompt_has_two_messages(self):
        prompt = self.build()
        assert len(prompt.messages) == 2

    def test_first_message_is_system(self):
        prompt = self.build()
        assert "system" in str(type(prompt.messages[0])).lower()

    def test_second_message_is_human(self):
        prompt = self.build()
        assert "human" in str(type(prompt.messages[1])).lower()

    def test_human_template_has_context_placeholder(self):
        assert "{context}" in self.human_template

    def test_human_template_has_question_placeholder(self):
        assert "{question}" in self.human_template

    def test_system_prompt_mentions_language_matching(self):
        """System prompt must instruct model to reply in user's language for multilingual support."""
        assert "языке" in self.system.lower()

    def test_system_prompt_mentions_context_grounding(self):
        """Model must be instructed to answer ONLY from context (avoid hallucinations)."""
        assert "контекст" in self.system.lower()

    def test_system_prompt_mentions_citations(self):
        """Source citations [1], [2] must be mentioned for traceability."""
        assert "[1]" in self.system or "cite" in self.system.lower()

    def test_system_prompt_asks_for_complete_answers(self):
        assert "законченными" in self.system.lower()
        assert "середине предложения" in self.system.lower()

    def test_empty_env_system_prompt_uses_default(self):
        with patch("src.generator.settings") as mock_settings:
            mock_settings.system_prompt = "   "
            assert self.get_system_prompt() == self.system

    def test_env_system_prompt_overrides_default(self):
        custom_prompt = "Отвечай как строгий RAG-ассистент."
        with patch("src.generator.settings") as mock_settings:
            mock_settings.system_prompt = custom_prompt
            assert self.get_system_prompt() == custom_prompt

    def test_prompt_uses_env_system_prompt(self):
        custom_prompt = "Отвечай только одним предложением."
        with patch("src.generator.settings") as mock_settings:
            mock_settings.system_prompt = custom_prompt
            prompt = self.build()

        formatted = prompt.format_messages(question="Что такое RAG?", context="RAG text")
        assert formatted[0].content == custom_prompt

    def test_prompt_formats_with_question_and_context(self):
        prompt = self.build()
        formatted = prompt.format_messages(
            question="What is Qdrant?",
            context="[1] (source: test.txt)\nQdrant is a vector database.",
        )
        combined = " ".join(str(m.content) for m in formatted)
        assert "What is Qdrant?" in combined
        assert "Qdrant is a vector database." in combined

    def test_prompt_formats_with_russian_question(self):
        prompt = self.build()
        formatted = prompt.format_messages(
            question="Что такое Qdrant?",
            context="[1] (source: ru.txt)\nQdrant — векторная база данных.",
        )
        combined = " ".join(str(m.content) for m in formatted)
        assert "Что такое Qdrant?" in combined
        assert "векторная" in combined


# ================================================================== Generator class

class TestGenerator:
    @pytest.fixture
    def mock_transformers(self):
        """Mock all HuggingFace and LangChain model loading."""
        with (
            patch("src.generator.AutoTokenizer") as mock_tokenizer_cls,
            patch("src.generator.hf_pipeline") as mock_pipeline,
            patch("src.generator.HuggingFacePipeline") as mock_hf_pipe_cls,
            patch("src.generator.ChatHuggingFace") as mock_chat_cls,
            patch("src.generator.torch") as mock_torch,
        ):
            mock_tokenizer = MagicMock()
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            mock_torch.cuda.is_available.return_value = False
            mock_torch.float32 = "float32"
            mock_torch.float16 = "float16"

            mock_pipeline.return_value = MagicMock()
            mock_hf_pipe_cls.return_value = MagicMock()

            mock_chat_instance = MagicMock()
            mock_chat_cls.return_value = mock_chat_instance

            yield {
                "tokenizer_cls": mock_tokenizer_cls,
                "pipeline": mock_pipeline,
                "hf_pipe_cls": mock_hf_pipe_cls,
                "chat_cls": mock_chat_cls,
                "chat_instance": mock_chat_instance,
                "torch": mock_torch,
            }

    def test_init_loads_tokenizer(self, mock_transformers):
        from src.generator import Generator
        Generator(model_name="Qwen/Qwen1.5-4B-Chat")
        mock_transformers["tokenizer_cls"].from_pretrained.assert_called_once_with(
            "Qwen/Qwen1.5-4B-Chat", trust_remote_code=True
        )

    def test_init_calls_hf_pipeline_with_model_name(self, mock_transformers):
        from src.generator import Generator
        Generator(model_name="Qwen/Qwen1.5-4B-Chat")
        call_kwargs = mock_transformers["pipeline"].call_args.kwargs
        assert call_kwargs.get("model") == "Qwen/Qwen1.5-4B-Chat"

    def test_init_creates_text_generation_pipeline(self, mock_transformers):
        from src.generator import Generator
        Generator(model_name="Qwen/Qwen1.5-4B-Chat")
        task_arg = mock_transformers["pipeline"].call_args.args[0]
        assert task_arg == "text-generation"

    def test_pipeline_uses_return_full_text_false(self, mock_transformers):
        """return_full_text=False ensures only new tokens are returned, not the full prompt."""
        from src.generator import Generator
        Generator(model_name="Qwen/Qwen1.5-4B-Chat")
        call_kwargs = mock_transformers["pipeline"].call_args.kwargs
        assert call_kwargs.get("return_full_text") is False

    def test_pipeline_removes_default_max_length_warning_source(self, mock_transformers):
        """max_length=20 must not survive alongside max_new_tokens."""
        mock_pipe = MagicMock()
        mock_pipe._forward_params = {"max_length": 20, "max_new_tokens": 384}
        mock_pipe.model = SimpleNamespace(generation_config=SimpleNamespace(max_length=20))
        mock_transformers["pipeline"].return_value = mock_pipe

        from src.generator import Generator
        with patch("src.generator.settings") as mock_settings:
            mock_settings.generator_model = "Qwen/Qwen1.5-4B-Chat"
            mock_settings.max_new_tokens = 384
            mock_settings.temperature = 0.0
            mock_settings.torch_dtype = "auto"
            Generator()

        assert "max_length" not in mock_pipe._forward_params
        assert mock_pipe._forward_params["max_new_tokens"] == 384
        assert mock_pipe._forward_params["do_sample"] is False
        assert mock_pipe.model.generation_config.max_length is None
        assert mock_pipe.model.generation_config.max_new_tokens == 384
        assert mock_pipe.model.generation_config.do_sample is False
        assert mock_pipe.model.generation_config._from_model_config is False

    def test_zero_temperature_disables_sampling_without_temperature_arg(self, mock_transformers):
        from src.generator import Generator

        with patch("src.generator.settings") as mock_settings:
            mock_settings.generator_model = "Qwen/Qwen1.5-1.8B-Chat"
            mock_settings.max_new_tokens = 384
            mock_settings.temperature = 0.0
            mock_settings.torch_dtype = "auto"
            Generator()

        call_kwargs = mock_transformers["pipeline"].call_args.kwargs
        assert call_kwargs.get("do_sample") is False
        assert "temperature" not in call_kwargs

    def test_cpu_path_uses_device_cpu_not_device_map(self, mock_transformers):
        """On CPU, device_map must NOT be set to avoid meta-tensor segfault."""
        mock_transformers["torch"].cuda.is_available.return_value = False
        from src.generator import Generator
        Generator(model_name="Qwen/Qwen1.5-1.8B-Chat")
        call_kwargs = mock_transformers["pipeline"].call_args.kwargs
        assert call_kwargs.get("device") == "cpu"
        assert "device_map" not in call_kwargs

    def test_cuda_path_uses_device_map_auto(self, mock_transformers):
        """On CUDA, device_map='auto' must be set (not device='cuda') to handle multi-GPU."""
        mock_transformers["torch"].cuda.is_available.return_value = True
        from src.generator import Generator
        Generator(model_name="Qwen/Qwen1.5-1.8B-Chat")
        call_kwargs = mock_transformers["pipeline"].call_args.kwargs
        assert call_kwargs.get("device_map") == "auto"
        assert "device" not in call_kwargs

    def test_uses_dtype_not_torch_dtype(self, mock_transformers):
        """Must use `dtype` (not deprecated `torch_dtype`) to avoid transformers warning."""
        from src.generator import Generator
        Generator(model_name="Qwen/Qwen1.5-1.8B-Chat")
        call_kwargs = mock_transformers["pipeline"].call_args.kwargs
        assert "dtype" in call_kwargs, "Should use `dtype` parameter"
        assert "torch_dtype" not in call_kwargs, "`torch_dtype` is deprecated — use `dtype`"

    def test_auto_dtype_uses_float32_on_cpu(self, mock_transformers):
        """CPU defaults to float32 for compatibility; CUDA uses lower precision."""
        from src.generator import Generator
        Generator(model_name="Qwen/Qwen1.5-1.8B-Chat")
        call_kwargs = mock_transformers["pipeline"].call_args.kwargs
        assert call_kwargs.get("dtype") == "float32"

    def test_explicit_float16_dtype_is_supported(self, mock_transformers):
        from src.generator import Generator
        with patch("src.generator.settings") as mock_settings:
            mock_settings.generator_model = "Qwen/Qwen1.5-1.8B-Chat"
            mock_settings.max_new_tokens = 512
            mock_settings.temperature = 0.7
            mock_settings.torch_dtype = "float16"
            Generator()
        call_kwargs = mock_transformers["pipeline"].call_args.kwargs
        assert call_kwargs.get("dtype") == "float16"

    def test_wraps_pipeline_with_chat_huggingface(self, mock_transformers):
        """ChatHuggingFace applies the tokenizer's chat template (ChatML for Qwen)."""
        from src.generator import Generator
        Generator(model_name="Qwen/Qwen1.5-4B-Chat")
        mock_transformers["chat_cls"].assert_called_once()

    def test_langchain_property_returns_chat_llm(self, mock_transformers):
        from src.generator import Generator
        gen = Generator(model_name="Qwen/Qwen1.5-4B-Chat")
        assert gen.langchain is mock_transformers["chat_instance"]

    def test_uses_settings_model_name_as_default(self, mock_transformers):
        from src.generator import Generator
        with patch("src.generator.settings") as mock_settings:
            mock_settings.generator_model = "Qwen/Qwen1.5-4B-Chat"
            mock_settings.max_new_tokens = 512
            mock_settings.temperature = 0.7
            Generator()
        mock_transformers["tokenizer_cls"].from_pretrained.assert_called_once_with(
            "Qwen/Qwen1.5-4B-Chat", trust_remote_code=True
        )
