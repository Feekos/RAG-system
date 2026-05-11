from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.ragas_evaluator import (
    RagasRunConfig,
    build_ragas_samples,
    load_ragas_config,
    load_testset,
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
                "rag": {"top_k": 3, "lazy_generator": False},
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
    assert config.lazy_generator is False
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

    pipeline.query.assert_called_once_with("What is RAG?", stream=False)
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
