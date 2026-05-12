"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document


# ------------------------------------------------------------------ text fixtures

ENGLISH_TEXT = """\
Artificial intelligence is a broad field of computer science. It focuses on building smart machines.
Machine learning is a subset of AI that uses statistical methods to enable machines to improve
with experience. Deep learning uses neural networks with many layers to learn from large amounts of data.

Natural language processing allows computers to understand and generate human language.
Large language models like GPT and Qwen are trained on vast amounts of text data.
They can answer questions, summarize documents, and generate creative content.

Vector databases store numerical representations of data called embeddings.
Similarity search in vector databases allows finding semantically similar content quickly.
Qdrant is a vector database written in Rust that supports filtering during search.
"""

RUSSIAN_TEXT = """\
Искусственный интеллект — широкая область информатики. Она сосредоточена на создании умных машин.
Машинное обучение — это подмножество ИИ, которое использует статистические методы для улучшения
с опытом. Глубокое обучение использует нейронные сети со многими слоями для обучения на больших данных.

Обработка естественного языка позволяет компьютерам понимать и генерировать человеческий язык.
Большие языковые модели, такие как GPT и Qwen, обучены на огромном количестве текстовых данных.
Они могут отвечать на вопросы, резюмировать документы и генерировать творческий контент.

Векторные базы данных хранят числовые представления данных, называемые эмбеддингами.
Поиск по сходству в векторных базах данных позволяет быстро находить семантически схожий контент.
Qdrant — векторная база данных, написанная на Rust, поддерживающая фильтрацию во время поиска.
"""

MIXED_TEXT = ENGLISH_TEXT + "\n\n" + RUSSIAN_TEXT


# ------------------------------------------------------------------ document fixtures

@pytest.fixture
def sample_docs_en() -> list[Document]:
    return [
        Document(page_content="Qdrant is a vector database written in Rust.", metadata={"source": "en.txt"}),
        Document(page_content="Octen-Embedding-0.6B supports more than 100 languages.", metadata={"source": "en.txt"}),
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "en.txt"}),
    ]


@pytest.fixture
def sample_docs_ru() -> list[Document]:
    return [
        Document(page_content="Qdrant — векторная база данных, написанная на Rust.", metadata={"source": "ru.txt"}),
        Document(page_content="Octen-Embedding-0.6B поддерживает более 100 языков.", metadata={"source": "ru.txt"}),
        Document(page_content="RAG сочетает поиск с генерацией текста.", metadata={"source": "ru.txt"}),
    ]


@pytest.fixture
def tmp_english_file(tmp_path):
    f = tmp_path / "english.txt"
    f.write_text(ENGLISH_TEXT, encoding="utf-8")
    return f


@pytest.fixture
def tmp_russian_file(tmp_path):
    f = tmp_path / "russian.txt"
    f.write_text(RUSSIAN_TEXT, encoding="utf-8")
    return f


@pytest.fixture
def tmp_mixed_file(tmp_path):
    f = tmp_path / "mixed.txt"
    f.write_text(MIXED_TEXT, encoding="utf-8")
    return f
