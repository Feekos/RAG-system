"""
Логика интерактивного чата RAG
"""

import argparse
import os

# Load .env BEFORE any src imports so HF_HOME is applied before model loading.
from dotenv import load_dotenv
load_dotenv()

if "HF_HOME" in os.environ:
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.environ["HF_HOME"])
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

_BANNER = """
================================================
  RAG ver. 0.3 | Qwen-4B + Qdrant + Octen-Embedding-0.6B
================================================
Система поддерживает следующие языки: EN, RU.
Комманды:  /stats  /clear  /quit
"""


def _run_interactive(pipeline, stream: bool) -> None:
    print(_BANNER)
    while True:
        try:
            question = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\Чат завершен!")
            break

        if not question:
            continue
        if question.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Чат завершен!")
            break
        if question.lower() == "/clear":
            pipeline.clear_history()
            print("Контекст диалога очищен.")
            continue
        if question.lower() == "/stats":
            import json
            print(json.dumps(pipeline.stats(), indent=2))
            continue

        print("\nАссистент: ", end="", flush=True)
        response = pipeline.query(question, stream=stream)
        if not stream:
            print(response.answer)

        if response.sources:
            print(f"\n[Источник] {', '.join(response.sources)}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG чат-интерфейс")
    parser.add_argument("--query", "-q", help="Единичный вопрос (Не интерактивный режим)")
    parser.add_argument("--stream", action="store_true", help="Поток генерируемых токенов")
    parser.add_argument("--top-k", type=int, default=None, help="Переопределение поиска top-k")
    args = parser.parse_args()

    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline.create(top_k=args.top_k)

    if args.query:
        response = pipeline.query(args.query, stream=args.stream)
        if not args.stream:
            print(response.answer)
        if response.sources:
            print(f"\n[Источник] {', '.join(response.sources)}")
    else:
        _run_interactive(pipeline, stream=args.stream)


if __name__ == "__main__":
    main()
