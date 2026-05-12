"""
Логика запуска оценки RAGAS для локального пайплайна RAG

Команды:
    python evaluate_ragas.py
    python evaluate_ragas.py --config eval/ragas_config.json
    python evaluate_ragas.py --testset eval/testset.jsonl --output-dir eval/results
    python evaluate_ragas.py --experiments
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if "HF_HOME" in os.environ:
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.environ["HF_HOME"])
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--config", help="Path to RAGAS JSON config")
    parser.add_argument("--testset", help="Override testset path from config")
    parser.add_argument("--output-dir", help="Override output directory from config")
    parser.add_argument("--metrics", nargs="+", help="Override metric names")
    parser.add_argument("--top-k", type=int, help="Override retriever top-k")
    parser.add_argument("--index-path", help="Path to documents for indexing before evaluation")
    parser.add_argument("--reset-index", action="store_true", help="Reset Qdrant collection before indexing")
    parser.add_argument("--experiments", action="store_true", help="Run experiment matrix from config")
    args = parser.parse_args()

    from src.ragas_evaluator import load_ragas_config, run_ragas_evaluation, run_ragas_experiments

    config = load_ragas_config(args.config)
    if args.testset:
        config.testset_path = Path(args.testset)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.metrics:
        config.metrics = args.metrics
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.index_path:
        config.index_path = Path(args.index_path)
    if args.reset_index:
        config.reset_index = True

    result = run_ragas_experiments(config) if args.experiments else run_ragas_evaluation(config)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
