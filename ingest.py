"""
Ingest documents into the RAG knowledge base.

Usage:
    python ingest.py data/documents/         # ingest entire directory
    python ingest.py path/to/file.pdf        # ingest single file
    python ingest.py --reset data/documents/ # clear collection first
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env BEFORE any src imports so that HF_HOME (and other env vars)
# are set before huggingface_hub / sentence-transformers read them.
from dotenv import load_dotenv
load_dotenv()

# Apply HF_HOME early — huggingface_hub reads it on first use, but
# sentence-transformers caches the home dir at import time in some versions.
if "HF_HOME" in os.environ:
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.environ["HF_HOME"])
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка документов в Qdrant")
    parser.add_argument("path", help="Файл или каталог для загрузки")
    parser.add_argument(
        "--reset", action="store_true", help="Удалите и воссоздайте коллекцию заново перед использованием"
    )
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Ошибка: путь не существует: {target}")
        sys.exit(1)

    from src.rag_pipeline import RAGPipeline

    if args.reset:
        from src.embeddings import EmbeddingModel
        from src.vector_store import VectorStore
        store = VectorStore(EmbeddingModel())
        store.delete_collection()

    pipeline = RAGPipeline.create(lazy_generator=True)

    if target.is_dir():
        count = pipeline.ingest_directory(str(target))
    else:
        count = pipeline.ingest_file(str(target))

    print(f"\nЗагрузка завершена. Чанков сохранено: {count}")
    print(f"Статистика: {pipeline.stats()}")


if __name__ == "__main__":
    main()
