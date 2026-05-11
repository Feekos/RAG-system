"""
Run RAGAS evaluation for the local RAG pipeline.

Usage:
    python evaluate_ragas.py
    python evaluate_ragas.py --config eval/ragas_config.json
    python evaluate_ragas.py --testset eval/testset.jsonl --output-dir eval/results
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
    args = parser.parse_args()

    from src.ragas_evaluator import load_ragas_config, run_ragas_evaluation

    config = load_ragas_config(args.config)
    if args.testset:
        config.testset_path = Path(args.testset)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.metrics:
        config.metrics = args.metrics
    if args.top_k is not None:
        config.top_k = args.top_k

    result = run_ragas_evaluation(config)
    print(json.dumps({k: str(v) if k != "summary" else v for k, v in result.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
