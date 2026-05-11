from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

from .config import settings
from .embeddings import EmbeddingModel
from .generator import Generator
from .rag_pipeline import RAGPipeline


DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


@dataclass
class RagasRunConfig:
    testset_path: Path
    output_dir: Path
    metrics: List[str]
    top_k: int | None = None
    lazy_generator: bool = True
    write_json: bool = True
    write_csv: bool = True
    write_markdown: bool = True


def load_ragas_config(path: str | Path | None = None) -> RagasRunConfig:
    config_path = Path(path or settings.ragas_config_path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))

    rag_cfg = raw.get("rag", {})
    outputs_cfg = raw.get("outputs", {})

    return RagasRunConfig(
        testset_path=Path(raw.get("testset_path", settings.ragas_testset_path)),
        output_dir=Path(raw.get("output_dir", settings.ragas_output_dir)),
        metrics=list(raw.get("metrics", DEFAULT_METRICS)),
        top_k=rag_cfg.get("top_k"),
        lazy_generator=bool(rag_cfg.get("lazy_generator", True)),
        write_json=bool(outputs_cfg.get("json", True)),
        write_csv=bool(outputs_cfg.get("csv", True)),
        write_markdown=bool(outputs_cfg.get("markdown", True)),
    )


def load_testset(path: str | Path) -> List[dict[str, Any]]:
    testset_path = Path(path)
    rows: List[dict[str, Any]] = []
    with testset_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item.get("question") or item.get("user_input")
            if not question:
                raise ValueError(f"{testset_path}:{line_no}: missing 'question'")
            reference = item.get("reference") or item.get("ground_truth") or item.get("answer")
            if not reference:
                raise ValueError(f"{testset_path}:{line_no}: missing 'reference'")
            rows.append(
                {
                    "user_input": str(question),
                    "reference": str(reference),
                    "reference_contexts": list(item.get("reference_contexts", [])),
                }
            )
    return rows


def build_ragas_samples(
    pipeline: RAGPipeline,
    testset_rows: Iterable[dict[str, Any]],
) -> List[dict[str, Any]]:
    samples: List[dict[str, Any]] = []
    for row in testset_rows:
        response = pipeline.query(row["user_input"], stream=False)
        retrieved_contexts = [doc.page_content for doc in response.retrieved_docs]
        sample = {
            "user_input": row["user_input"],
            "response": response.answer,
            "retrieved_contexts": retrieved_contexts,
            "reference": row["reference"],
        }
        if row.get("reference_contexts"):
            sample["reference_contexts"] = row["reference_contexts"]
        samples.append(sample)
    return samples


def run_ragas_evaluation(
    config: RagasRunConfig,
    pipeline: RAGPipeline | None = None,
    evaluator_llm: Any | None = None,
    evaluator_embeddings: Any | None = None,
) -> dict[str, Path | dict[str, float]]:
    evaluate, Dataset = _import_ragas_runtime()
    rows = load_testset(config.testset_path)
    pipeline = pipeline or RAGPipeline.create(
        lazy_generator=config.lazy_generator,
        top_k=config.top_k,
    )
    samples = build_ragas_samples(pipeline, rows)

    evaluator_llm = evaluator_llm or _build_evaluator_llm()
    evaluator_embeddings = evaluator_embeddings or _build_evaluator_embeddings()
    metrics = _build_metrics(config.metrics, evaluator_llm, evaluator_embeddings)

    dataset = Dataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    dataframe = result.to_pandas()
    result_rows = _to_jsonable(dataframe.to_dict(orient="records"))
    summary = _summarize_scores(result_rows)
    paths = write_ragas_outputs(config, samples, result_rows, summary)
    return {"summary": summary, **paths}


def write_ragas_outputs(
    config: RagasRunConfig,
    samples: List[dict[str, Any]],
    result_rows: List[dict[str, Any]],
    summary: dict[str, float],
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written: dict[str, Path] = {}

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": config.metrics,
        "summary": summary,
        "samples": samples,
        "results": result_rows,
    }

    if config.write_json:
        path = config.output_dir / f"ragas_{stamp}.json"
        path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        written["json"] = path

    if config.write_csv:
        path = config.output_dir / f"ragas_{stamp}.csv"
        _write_csv(path, result_rows)
        written["csv"] = path

    if config.write_markdown:
        path = config.output_dir / f"ragas_{stamp}.md"
        path.write_text(_render_markdown(summary, result_rows), encoding="utf-8")
        written["markdown"] = path

    return written


def _import_ragas_runtime():
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return evaluate, Dataset


def _build_evaluator_llm():
    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RuntimeError("Installed RAGAS version does not expose LangchainLLMWrapper.") from exc
    return LangchainLLMWrapper(Generator().langchain)


def _build_evaluator_embeddings():
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        raise RuntimeError(
            "Installed RAGAS version does not expose LangchainEmbeddingsWrapper."
        ) from exc
    return LangchainEmbeddingsWrapper(EmbeddingModel().langchain)


def _build_metrics(metric_names: List[str], evaluator_llm: Any, evaluator_embeddings: Any) -> List[Any]:
    from ragas import metrics as ragas_metrics

    built: List[Any] = []
    for name in metric_names:
        normalized = name.strip().lower()
        if normalized == "faithfulness":
            built.append(_metric_from_candidates(ragas_metrics, ["Faithfulness", "faithfulness"], llm=evaluator_llm))
        elif normalized == "answer_relevancy":
            built.append(
                _metric_from_candidates(
                    ragas_metrics,
                    ["ResponseRelevancy", "AnswerRelevancy", "answer_relevancy"],
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                )
            )
        elif normalized == "context_precision":
            built.append(
                _metric_from_candidates(
                    ragas_metrics,
                    [
                        "LLMContextPrecisionWithReference",
                        "ContextPrecision",
                        "context_precision",
                    ],
                    llm=evaluator_llm,
                )
            )
        elif normalized == "context_recall":
            built.append(
                _metric_from_candidates(
                    ragas_metrics,
                    ["LLMContextRecall", "ContextRecall", "context_recall"],
                    llm=evaluator_llm,
                )
            )
        else:
            raise ValueError(f"Unsupported RAGAS metric: {name}")
    return built


def _metric_from_candidates(module: Any, candidates: List[str], **kwargs: Any) -> Any:
    for candidate in candidates:
        metric = getattr(module, candidate, None)
        if metric is None:
            continue
        if isinstance(metric, type):
            accepted = _accepted_kwargs(metric, kwargs)
            return metric(**accepted)
        for attr, value in kwargs.items():
            if hasattr(metric, attr):
                setattr(metric, attr, value)
        return metric
    raise RuntimeError(f"None of these RAGAS metrics are available: {', '.join(candidates)}")


def _accepted_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        import inspect

        signature = inspect.signature(cls)
        return {key: value for key, value in kwargs.items() if key in signature.parameters}
    except (TypeError, ValueError):
        return kwargs


def _summarize_scores(rows: List[dict[str, Any]]) -> dict[str, float]:
    numeric: dict[str, List[float]] = {}
    excluded = {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts"}
    for row in rows:
        for key, value in row.items():
            if key in excluded:
                continue
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in numeric.items() if values}


def _write_csv(path: Path, rows: List[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(summary: dict[str, float], rows: List[dict[str, Any]]) -> str:
    lines = ["# RAGAS Evaluation Results", ""]
    lines.append("## Summary")
    lines.append("")
    if summary:
        lines.append("| Metric | Score |")
        lines.append("|---|---:|")
        for key, value in summary.items():
            lines.append(f"| `{key}` | {value:.4f} |")
    else:
        lines.append("No numeric metric scores were produced.")

    lines.append("")
    lines.append("## Samples")
    lines.append("")
    for index, row in enumerate(rows, 1):
        question = row.get("user_input", "")
        lines.append(f"### {index}. {question}")
        for key, value in row.items():
            if key == "user_input":
                continue
            if isinstance(value, float):
                lines.append(f"- `{key}`: {value:.4f}")
            elif isinstance(value, (str, int)):
                lines.append(f"- `{key}`: {value}")
        lines.append("")
    return "\n".join(lines)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
