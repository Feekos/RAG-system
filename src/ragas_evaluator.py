from __future__ import annotations

import csv
import inspect
import itertools
import json
import math
import os
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

from .config import settings
from .embeddings import EmbeddingModel
from .rag_pipeline import RAGPipeline


DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "context_entities_recall",
    "noise_sensitivity",
    "semantic_similarity",
    "factual_correctness",
    "answer_accuracy",
    "context_relevance",
    "response_groundedness",
]


@dataclass
class RagasRunConfig:
    testset_path: Path
    output_dir: Path
    metrics: List[str]
    top_k: int | None = None
    index_path: Path | None = None
    reset_index: bool = False
    lazy_generator: bool = True
    write_json: bool = True
    write_csv: bool = True
    write_markdown: bool = True
    run_name: str | None = None
    metadata: dict[str, Any] | None = None
    experiments: dict[str, List[Any]] | None = None


def load_ragas_config(path: str | Path | None = None) -> RagasRunConfig:
    config_path = Path(path or settings.ragas_config_path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))

    rag_cfg = raw.get("rag", {})
    outputs_cfg = raw.get("outputs", {})
    experiments_cfg = raw.get("experiments", {})

    return RagasRunConfig(
        testset_path=Path(raw.get("testset_path", settings.ragas_testset_path)),
        output_dir=Path(raw.get("output_dir", settings.ragas_output_dir)),
        metrics=list(raw.get("metrics", DEFAULT_METRICS)),
        top_k=rag_cfg.get("top_k"),
        index_path=Path(rag_cfg["index_path"]) if rag_cfg.get("index_path") else None,
        reset_index=bool(rag_cfg.get("reset_index", settings.ragas_reset_index)),
        lazy_generator=bool(rag_cfg.get("lazy_generator", True)),
        write_json=bool(outputs_cfg.get("json", True)),
        write_csv=bool(outputs_cfg.get("csv", True)),
        write_markdown=bool(outputs_cfg.get("markdown", True)),
        experiments=_normalize_experiment_matrix(experiments_cfg),
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
        response = pipeline.query(row["user_input"], stream=False, use_history=False)
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
    evaluator_llm = evaluator_llm or _build_evaluator_llm()
    evaluator_embeddings = evaluator_embeddings or _build_evaluator_embeddings()
    metrics = _build_metrics(config.metrics, evaluator_llm, evaluator_embeddings)

    with _temporary_env({"RAGAS_RESET_INDEX": "true"} if config.reset_index else {}):
        pipeline = pipeline or RAGPipeline.create(
            lazy_generator=config.lazy_generator,
            top_k=config.top_k,
        )
        if config.index_path is not None:
            if config.reset_index:
                pipeline.reset_collection()
                pipeline = RAGPipeline.create(
                    lazy_generator=config.lazy_generator,
                    top_k=config.top_k,
                )
            target = config.index_path
            if target.is_dir():
                pipeline.ingest_directory(str(target))
            else:
                pipeline.ingest_file(str(target))
    samples = build_ragas_samples(pipeline, rows)

    dataset = Dataset.from_list(samples)
    print(
        "[RAGAS] Evaluation settings: "
        f"timeout={settings.ragas_timeout}s, "
        f"max_workers={settings.ragas_max_workers}, "
        f"llm_timeout={settings.ragas_llm_timeout}s, "
        f"llm_max_tokens={settings.ragas_llm_max_tokens}"
    )
    evaluate_kwargs = {
        "dataset": dataset,
        "metrics": metrics,
        "llm": evaluator_llm,
        "embeddings": evaluator_embeddings,
    }
    run_config = _build_ragas_run_config()
    if run_config is not None and "run_config" in inspect.signature(evaluate).parameters:
        evaluate_kwargs["run_config"] = run_config
    if "raise_exceptions" in inspect.signature(evaluate).parameters:
        evaluate_kwargs["raise_exceptions"] = False
    result = evaluate(**evaluate_kwargs)

    dataframe = result.to_pandas()
    result_rows = _to_jsonable(dataframe.to_dict(orient="records"))
    summary, summary_counts = _summarize_scores(result_rows)
    paths = write_ragas_outputs(config, samples, result_rows, summary)
    return {"summary": summary, "summary_counts": summary_counts, **paths}


def run_ragas_experiments(config: RagasRunConfig) -> dict[str, Any]:
    experiments = list(_iter_experiments(config.experiments or {}))
    if not experiments:
        return run_ragas_evaluation(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    leaderboard: list[dict[str, Any]] = []

    for index, overrides in enumerate(experiments, 1):
        run_name = _format_run_name(index, overrides)
        run_config = _copy_config_for_experiment(config, run_name, overrides)
        settings_overrides, top_k = _split_experiment_overrides(overrides)
        if top_k is not None:
            run_config.top_k = top_k

        print(f"[RAGAS] Experiment {index}/{len(experiments)}: {run_name}")
        with _temporary_settings(settings_overrides), _temporary_env(
            {"RAGAS_RESET_INDEX": "true"} if run_config.reset_index else {}
        ):
            result = run_ragas_evaluation(run_config)

        row = {
            "run_name": run_name,
            **overrides,
            **{f"score_{key}": value for key, value in result["summary"].items()},
        }
        for key, value in result.items():
            if key != "summary":
                row[f"path_{key}"] = str(value)
        leaderboard.append(row)

    leaderboard_path = config.output_dir / f"ragas_experiments_{stamp}.csv"
    _write_csv(leaderboard_path, leaderboard)
    summary_path = config.output_dir / f"ragas_experiments_{stamp}.md"
    summary_path.write_text(_render_leaderboard_markdown(leaderboard), encoding="utf-8")
    return {"leaderboard_csv": leaderboard_path, "leaderboard_markdown": summary_path, "runs": leaderboard}


def write_ragas_outputs(
    config: RagasRunConfig,
    samples: List[dict[str, Any]],
    result_rows: List[dict[str, Any]],
    summary: dict[str, float],
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"ragas_{_slug(config.run_name)}_{stamp}" if config.run_name else f"ragas_{stamp}"
    written: dict[str, Path] = {}

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": config.metrics,
        "run_name": config.run_name,
        "metadata": config.metadata or {},
        "summary": summary,
        "samples": samples,
        "results": result_rows,
    }

    if config.write_json:
        path = config.output_dir / f"{prefix}.json"
        path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        written["json"] = path

    if config.write_csv:
        path = config.output_dir / f"{prefix}.csv"
        _write_csv(path, result_rows)
        written["csv"] = path

    if config.write_markdown:
        path = config.output_dir / f"{prefix}.md"
        path.write_text(_render_markdown(summary, result_rows, config), encoding="utf-8")
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
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluator LLM uses a vLLM/OpenAI-compatible endpoint. "
            "Install langchain-openai with: pip install -r requirements.txt"
        ) from exc

    llm = ChatOpenAI(
        model=settings.ragas_llm_model,
        base_url=settings.ragas_llm_base_url,
        api_key=settings.ragas_llm_api_key,
        temperature=settings.ragas_llm_temperature,
        max_tokens=settings.ragas_llm_max_tokens,
        timeout=settings.ragas_llm_timeout,
    )
    _ensure_openai_compatible_endpoint(settings.ragas_llm_base_url, settings.ragas_llm_api_key)
    return LangchainLLMWrapper(llm)


def _build_ragas_run_config() -> Any | None:
    try:
        from ragas.run_config import RunConfig
    except ImportError:
        try:
            from ragas import RunConfig
        except ImportError:
            return None

    kwargs = {
        "timeout": settings.ragas_timeout,
        "max_workers": settings.ragas_max_workers,
    }
    accepted = _accepted_kwargs(RunConfig, kwargs)
    return RunConfig(**accepted)


def _ensure_openai_compatible_endpoint(base_url: str, api_key: str) -> None:
    models_url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(
        models_url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    deadline = time.monotonic() + settings.ragas_llm_wait_timeout
    last_error: BaseException | None = None
    print(f"[RAGAS] Waiting for evaluator LLM endpoint: {models_url}")

    while time.monotonic() <= deadline:
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status < 400:
                    print("[RAGAS] Evaluator LLM endpoint is ready.")
                    return
                last_error = RuntimeError(f"HTTP {response.status}")
        except (OSError, TimeoutError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
        time.sleep(settings.ragas_llm_wait_interval)

    if last_error is not None:
        raise RuntimeError(
            "RAGAS evaluator LLM endpoint is not available: "
            f"{models_url}. Start vLLM before running evaluation:\n"
            "  docker compose --profile eval run --rm rag-eval\n"
            "If vLLM is already starting, wait until the model is loaded. Check it with:\n"
            "  curl -H 'Authorization: Bearer local-vllm-key' http://localhost:8001/v1/models\n"
            "and inspect logs with:\n"
            "  docker compose logs -f vllm"
        ) from last_error


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
        elif normalized == "context_entities_recall":
            built.append(
                _metric_from_candidates(
                    ragas_metrics,
                    [
                        "ContextEntityRecall",
                        "ContextEntitiesRecall",
                        "context_entity_recall",
                        "context_entities_recall",
                    ],
                    llm=evaluator_llm,
                )
            )
        elif normalized == "noise_sensitivity":
            built.append(
                _metric_from_candidates(
                    ragas_metrics,
                    ["NoiseSensitivity", "noise_sensitivity"],
                    llm=evaluator_llm,
                )
            )
        elif normalized == "semantic_similarity":
            built.append(
                _metric_from_candidates(
                    ragas_metrics,
                    [
                        "SemanticSimilarity",
                        "AnswerSimilarity",
                        "answer_similarity",
                        "semantic_similarity",
                    ],
                    embeddings=evaluator_embeddings,
                )
            )
        elif normalized == "factual_correctness":
            built.append(
                _metric_from_candidates(
                    ragas_metrics,
                    ["FactualCorrectness", "AnswerCorrectness", "answer_correctness", "factual_correctness"],
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                )
            )
        elif normalized == "answer_accuracy":
            built.append(
                _metric_from_candidates(
                    ragas_metrics,
                    ["AnswerAccuracy", "answer_accuracy"],
                    llm=evaluator_llm,
                )
            )
        elif normalized == "context_relevance":
            built.append(
                _metric_from_candidates(
                    ragas_metrics,
                    ["ContextRelevance", "ContextRelevancy", "context_relevance", "context_relevancy"],
                    llm=evaluator_llm,
                )
            )
        elif normalized == "response_groundedness":
            built.append(
                _metric_from_candidates(
                    ragas_metrics,
                    ["ResponseGroundedness", "response_groundedness"],
                    llm=evaluator_llm,
                )
            )
        else:
            raise ValueError(f"Unsupported RAGAS metric: {name}")
    for metric in built:
        _set_metric_timeout(metric, settings.ragas_timeout)
    return built


def _set_metric_timeout(metric: Any, timeout: int) -> None:
    for attr in ("timeout", "_timeout"):
        if hasattr(metric, attr):
            try:
                setattr(metric, attr, timeout)
            except (AttributeError, TypeError):
                pass


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


def _summarize_scores(rows: List[dict[str, Any]]) -> tuple[dict[str, float], dict[str, int]]:
    numeric: dict[str, List[float]] = {}
    excluded = {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts"}
    for row in rows:
        for key, value in row.items():
            if key in excluded:
                continue
            if isinstance(value, (int, float)):
                number = float(value)
                if math.isfinite(number):
                    numeric.setdefault(key, []).append(number)
    summary = {key: sum(values) / len(values) for key, values in numeric.items() if values}
    counts = {key: len(values) for key, values in numeric.items() if values}
    return summary, counts


def _write_csv(path: Path, rows: List[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(
    summary: dict[str, float],
    rows: List[dict[str, Any]],
    config: RagasRunConfig | None = None,
) -> str:
    lines = ["# RAGAS Evaluation Results", ""]
    if config and (config.run_name or config.metadata):
        lines.append("## Run")
        lines.append("")
        if config.run_name:
            lines.append(f"- `run_name`: {config.run_name}")
        for key, value in (config.metadata or {}).items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
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


def _normalize_experiment_matrix(raw: Any) -> dict[str, List[Any]] | None:
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("'experiments' must be an object of parameter lists.")
    matrix: dict[str, List[Any]] = {}
    for key, value in raw.items():
        if value is None:
            continue
        matrix[key] = value if isinstance(value, list) else [value]
    return matrix


def _iter_experiments(matrix: dict[str, List[Any]]) -> Iterable[dict[str, Any]]:
    if not matrix:
        return []
    keys = list(matrix.keys())
    values = [matrix[key] for key in keys]
    return (dict(zip(keys, combination)) for combination in itertools.product(*values))


def _copy_config_for_experiment(
    config: RagasRunConfig,
    run_name: str,
    overrides: dict[str, Any],
) -> RagasRunConfig:
    return RagasRunConfig(
        testset_path=config.testset_path,
        output_dir=config.output_dir,
        metrics=list(config.metrics),
        top_k=config.top_k,
        index_path=config.index_path,
        reset_index=config.reset_index,
        lazy_generator=config.lazy_generator,
        write_json=config.write_json,
        write_csv=config.write_csv,
        write_markdown=config.write_markdown,
        run_name=run_name,
        metadata=overrides,
        experiments=None,
    )


def _split_experiment_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    top_k = overrides.get("top_k")
    allowed_settings = {
        "embedding_model",
        "embedding_dim",
        "generator_model",
        "max_new_tokens",
        "temperature",
        "torch_dtype",
        "chunk_size",
        "chunk_overlap",
        "qdrant_collection",
    }
    return (
        {key: value for key, value in overrides.items() if key in allowed_settings},
        int(top_k) if top_k is not None else None,
    )


@contextmanager
def _temporary_settings(overrides: dict[str, Any]):
    previous = {key: getattr(settings, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(settings, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(settings, key, value)


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _format_run_name(index: int, overrides: dict[str, Any]) -> str:
    parts = [f"{key}-{value}" for key, value in sorted(overrides.items())]
    return f"{index:03d}_" + "_".join(parts)


def _slug(value: str | None) -> str:
    if not value:
        return "run"
    chars = [char if char.isalnum() else "_" for char in value.lower()]
    return "_".join("".join(chars).split("_"))[:120]


def _render_leaderboard_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# RAGAS Experiment Leaderboard", ""]
    if not rows:
        lines.append("No experiment rows were produced.")
        return "\n".join(lines)

    columns = list(dict.fromkeys(key for row in rows for key in row.keys()))
    score_columns = [column for column in columns if column.startswith("score_")]
    display_columns = ["run_name"] + [
        column for column in columns if column not in {"run_name"} and not column.startswith("path_")
    ]
    if score_columns:
        display_columns = ["run_name"] + score_columns + [
            column
            for column in display_columns
            if column != "run_name" and column not in score_columns
        ]

    lines.append("| " + " | ".join(display_columns) + " |")
    lines.append("|" + "|".join("---" for _ in display_columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in display_columns) + " |")
    return "\n".join(lines)
