from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.ragas_evaluator import (
    RagasRunConfig,
    _build_evaluator_llm,
    build_ragas_samples,
    load_ragas_config,
    load_testset,
    run_ragas_evaluation,
    run_ragas_experiments,
    write_ragas_outputs,
)


def test_load_ragas_config_reads_json(tmp_path):
    config_path = tmp_path / "ragas_config.json"
    config_path.write_text(
        json.dumps(
            {
                "testset_path": "eval/custom.jsonl",
                "output_dir": "eval/out",
                "metrics": ["faithfulness"],
                "rag": {
                    "top_k": 3,
                    "lazy_generator": False,
                    "index_path": "data/documents",
                    "reset_index": True,
                },
                "experiments": {"top_k": [3, 5], "chunk_size": [384]},
                "outputs": {"json": True, "csv": False, "markdown": True},
            }
        ),
        encoding="utf-8",
    )

    config = load_ragas_config(config_path)

    assert config.testset_path == Path("eval/custom.jsonl")
    assert config.output_dir == Path("eval/out")
    assert config.metrics == ["faithfulness"]
    assert config.top_k == 3
    assert config.index_path == Path("data/documents")
    assert config.reset_index is True
    assert config.lazy_generator is False
    assert config.experiments == {"top_k": [3, 5], "chunk_size": [384]}
    assert config.write_csv is False


def test_load_testset_accepts_jsonl(tmp_path):
    testset = tmp_path / "testset.jsonl"
    testset.write_text(
        '{"question":"What is RAG?","reference":"Retrieval plus generation."}\n',
        encoding="utf-8",
    )

    rows = load_testset(testset)

    assert rows == [
        {
            "user_input": "What is RAG?",
            "reference": "Retrieval plus generation.",
            "reference_contexts": [],
        }
    ]


def test_build_ragas_samples_calls_pipeline_query():
    pipeline = MagicMock()
    response = MagicMock()
    response.answer = "RAG combines retrieval and generation."
    response.retrieved_docs = [
        Document(page_content="RAG retrieves context.", metadata={"source": "sample.txt"})
    ]
    pipeline.query.return_value = response

    samples = build_ragas_samples(
        pipeline,
        [{"user_input": "What is RAG?", "reference": "Reference answer."}],
    )

    pipeline.query.assert_called_once_with("What is RAG?", stream=False, use_history=False)
    assert samples[0]["response"] == "RAG combines retrieval and generation."
    assert samples[0]["retrieved_contexts"] == ["RAG retrieves context."]
    assert samples[0]["reference"] == "Reference answer."


def test_write_ragas_outputs_creates_json_csv_and_markdown(tmp_path):
    config = RagasRunConfig(
        testset_path=tmp_path / "testset.jsonl",
        output_dir=tmp_path / "results",
        metrics=["faithfulness"],
    )

    written = write_ragas_outputs(
        config,
        samples=[{"user_input": "q", "response": "a", "retrieved_contexts": [], "reference": "r"}],
        result_rows=[{"user_input": "q", "faithfulness": 0.9}],
        summary={"faithfulness": 0.9},
    )

    assert set(written) == {"json", "csv", "markdown"}
    assert written["json"].exists()
    assert written["csv"].exists()
    assert written["markdown"].exists()
    assert "faithfulness" in written["markdown"].read_text(encoding="utf-8")


def test_run_ragas_experiments_writes_leaderboard(tmp_path):
    config = RagasRunConfig(
        testset_path=tmp_path / "testset.jsonl",
        output_dir=tmp_path / "results",
        metrics=["faithfulness"],
        experiments={"top_k": [3, 5]},
    )

    with patch("src.ragas_evaluator.run_ragas_evaluation") as mock_run:
        mock_run.return_value = {
            "summary": {"faithfulness": 0.8},
            "json": tmp_path / "result.json",
        }
        result = run_ragas_experiments(config)

    assert mock_run.call_count == 2
    assert result["leaderboard_csv"].exists()
    assert result["leaderboard_markdown"].exists()


def test_run_ragas_evaluation_does_not_raise_on_single_metric_failure(tmp_path):
    testset = tmp_path / "testset.jsonl"
    testset.write_text(
        '{"question":"What is RAG?","reference":"Retrieval plus generation."}\n',
        encoding="utf-8",
    )
    config = RagasRunConfig(
        testset_path=testset,
        output_dir=tmp_path / "results",
        metrics=[],
        write_csv=False,
        write_markdown=False,
    )
    pipeline = MagicMock()
    response = MagicMock()
    response.answer = "RAG combines retrieval and generation."
    response.retrieved_docs = []
    pipeline.query.return_value = response

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeResult:
        def to_pandas(self):
            return MagicMock(to_dict=MagicMock(return_value=[{"faithfulness": None}]))

    def fake_evaluate(**kwargs):
        assert kwargs["raise_exceptions"] is False
        return FakeResult()

    with (
        patch("src.ragas_evaluator._import_ragas_runtime", return_value=(fake_evaluate, FakeDataset)),
        patch("src.ragas_evaluator._build_evaluator_llm", return_value=MagicMock()),
        patch("src.ragas_evaluator._build_evaluator_embeddings", return_value=MagicMock()),
        patch("src.ragas_evaluator._build_metrics", return_value=[]),
    ):
        result = run_ragas_evaluation(config, pipeline=pipeline)

    assert result["json"].exists()


def test_build_evaluator_llm_uses_openai_compatible_endpoint():
    created = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            created["chat_kwargs"] = kwargs

    class FakeWrapper:
        def __init__(self, llm):
            created["wrapped_llm"] = llm

    fake_ragas_llms = SimpleNamespace(LangchainLLMWrapper=FakeWrapper)
    fake_langchain_openai = SimpleNamespace(ChatOpenAI=FakeChatOpenAI)
    response = MagicMock()
    response.__enter__.return_value.status = 200

    with patch.dict(
        sys.modules,
        {
            "ragas.llms": fake_ragas_llms,
            "langchain_openai": fake_langchain_openai,
        },
    ), patch("src.ragas_evaluator.urllib.request.urlopen", return_value=response) as mock_urlopen:
        wrapper = _build_evaluator_llm()

    assert isinstance(wrapper, FakeWrapper)
    assert isinstance(created["wrapped_llm"], FakeChatOpenAI)
    assert created["chat_kwargs"]["base_url"]
    assert created["chat_kwargs"]["api_key"]
    assert created["chat_kwargs"]["model"]
    mock_urlopen.assert_called_once()
