"""Tests for RAGPipeline — orchestration, LCEL chain, multilingual queries."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag_pipeline import RAGPipeline, RAGResponse


# ================================================================== fixtures

@pytest.fixture
def mock_processor():
    mock = MagicMock()
    mock.load_file.return_value = [
        Document(page_content="Chunk A about Qdrant.", metadata={"source": "doc.txt"}),
        Document(page_content="Chunk B about RAG.", metadata={"source": "doc.txt"}),
    ]
    mock.load_directory.return_value = [
        Document(page_content="Dir chunk.", metadata={"source": "dir.txt"}),
    ]
    return mock


@pytest.fixture
def mock_store():
    mock = MagicMock()
    mock.add_documents.return_value = ["id1", "id2"]
    mock.count.return_value = 10
    mock.as_retriever.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_retriever():
    mock = MagicMock()
    mock.retrieve_with_context.return_value = (
        [Document(page_content="Qdrant stores vectors.", metadata={"source": "doc.txt"})],
        "[1] (source: doc.txt)\nQdrant stores vectors.",
    )
    return mock


@pytest.fixture
def mock_generator():
    mock = MagicMock()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Qdrant is a vector database [1]."
    mock_chain.stream.return_value = iter(["Qdrant ", "is ", "a vector database [1]."])
    mock.langchain = MagicMock()
    return mock, mock_chain


@pytest.fixture
def pipeline(mock_processor, mock_store, mock_retriever, mock_generator):
    gen, chain = mock_generator
    p = RAGPipeline(
        processor=mock_processor,
        store=mock_store,
        retriever=mock_retriever,
        generator=gen,
    )
    p._gen_chain = chain
    return p, mock_processor, mock_store, mock_retriever, gen, chain


# ================================================================== RAGResponse

class TestRAGResponse:
    def test_sources_extracts_unique_sources(self):
        docs = [
            Document(page_content="A", metadata={"source": "a.txt"}),
            Document(page_content="B", metadata={"source": "a.txt"}),
            Document(page_content="C", metadata={"source": "b.txt"}),
        ]
        resp = RAGResponse(answer="ans", question="q", retrieved_docs=docs)
        assert resp.sources == ["a.txt", "b.txt"]

    def test_sources_empty_when_no_docs(self):
        resp = RAGResponse(answer="ans", question="q")
        assert resp.sources == []


# ================================================================== query

class TestQuery:
    def test_query_returns_rag_response(self, pipeline):
        p, *_ = pipeline
        response = p.query("What is Qdrant?")
        assert isinstance(response, RAGResponse)

    def test_query_answer_content(self, pipeline):
        p, _, _, _, _, chain = pipeline
        response = p.query("What is Qdrant?")
        assert response.answer == "Qdrant is a vector database [1]."

    def test_query_includes_retrieved_docs(self, pipeline):
        p, *_ = pipeline
        response = p.query("What is Qdrant?")
        assert len(response.retrieved_docs) == 1
        assert response.retrieved_docs[0].page_content == "Qdrant stores vectors."

    def test_query_passes_question_to_chain(self, pipeline):
        p, _, _, _, _, chain = pipeline
        p.query("What is Qdrant?")
        call_kwargs = chain.invoke.call_args[0][0]
        assert call_kwargs["question"] == "What is Qdrant?"

    def test_query_passes_context_to_chain(self, pipeline):
        p, _, _, _, _, chain = pipeline
        p.query("What is Qdrant?")
        call_kwargs = chain.invoke.call_args[0][0]
        assert "Qdrant stores vectors." in call_kwargs["context"]

    def test_query_russian_question(self, pipeline):
        p, _, _, mock_retriever, _, chain = pipeline
        mock_retriever.retrieve_with_context.return_value = (
            [Document(page_content="Qdrant — векторная база.", metadata={"source": "ru.txt"})],
            "[1] (source: ru.txt)\nQdrant — векторная база.",
        )
        chain.invoke.return_value = "Qdrant — это векторная база данных [1]."
        response = p.query("Что такое Qdrant?")
        assert response.question == "Что такое Qdrant?"
        assert response.answer == "Qdrant — это векторная база данных [1]."

    def test_query_empty_retrieval_returns_no_docs_message(self, pipeline):
        p, _, _, mock_retriever, _, _ = pipeline
        mock_retriever.retrieve_with_context.return_value = ([], "No relevant context found.")
        response = p.query("unknown topic")
        assert "No relevant documents found" in response.answer
        assert response.retrieved_docs == []

    def test_query_lazy_loads_generator_on_first_call(self, mock_processor, mock_store, mock_retriever):
        """Pipeline with lazy_generator should load the LLM only on first query."""
        pipeline = RAGPipeline(
            processor=mock_processor,
            store=mock_store,
            retriever=mock_retriever,
            generator=None,
        )
        # _gen_chain must be None before any query
        assert pipeline._gen_chain is None

    def test_query_stream_concatenates_chunks(self, pipeline):
        p, _, _, _, _, chain = pipeline
        chain.stream.return_value = iter(["Hello ", "world!"])
        response = p.query("test", stream=True)
        assert "Hello" in response.answer
        assert "world" in response.answer


# ================================================================== ingest

class TestIngest:
    def test_ingest_file_calls_processor_load_file(self, pipeline):
        p, mock_processor, *_ = pipeline
        p.ingest_file("path/to/doc.txt")
        mock_processor.load_file.assert_called_once_with("path/to/doc.txt")

    def test_ingest_file_adds_documents_to_store(self, pipeline):
        p, _, mock_store, *_ = pipeline
        count = p.ingest_file("path/to/doc.txt")
        mock_store.add_documents.assert_called()
        assert count == 2  # mock_processor returns 2 chunks

    def test_ingest_directory_calls_processor_load_directory(self, pipeline):
        p, mock_processor, *_ = pipeline
        p.ingest_directory("data/documents/")
        mock_processor.load_directory.assert_called_once_with("data/documents/")

    def test_ingest_empty_docs_returns_zero(self, pipeline):
        p, mock_processor, mock_store, *_ = pipeline
        mock_processor.load_file.return_value = []
        count = p.ingest_file("empty.txt")
        mock_store.add_documents.assert_not_called()
        assert count == 0

    def test_ingest_batches_large_documents(self, mock_processor, mock_store, mock_retriever):
        """Verify batching: 150 docs with batch_size=64 → 3 add_documents calls."""
        big_docs = [Document(page_content=f"Chunk {i}", metadata={"source": "big.txt"}) for i in range(150)]
        mock_processor.load_file.return_value = big_docs

        p = RAGPipeline(processor=mock_processor, store=mock_store, retriever=mock_retriever)
        p.ingest_file("big.txt", batch_size=64)

        assert mock_store.add_documents.call_count == 3  # ceil(150/64) = 3


# ================================================================== stats

class TestStats:
    def test_stats_returns_dict(self, pipeline):
        p, *_ = pipeline
        with patch("src.rag_pipeline.settings") as mock_settings:
            mock_settings.qdrant_collection = "documents"
            mock_settings.embedding_model = "BAAI/bge-m3"
            mock_settings.generator_model = "Qwen/Qwen1.5-4B-Chat"
            result = p.stats()
        assert isinstance(result, dict)

    def test_stats_contains_required_keys(self, pipeline):
        p, *_ = pipeline
        with patch("src.rag_pipeline.settings") as mock_settings:
            mock_settings.qdrant_collection = "documents"
            mock_settings.embedding_model = "BAAI/bge-m3"
            mock_settings.generator_model = "Qwen/Qwen1.5-4B-Chat"
            result = p.stats()
        assert "collection" in result
        assert "total_chunks" in result
        assert "embedding_model" in result
        assert "generator_model" in result

    def test_stats_total_chunks_from_store(self, pipeline):
        p, _, mock_store, *_ = pipeline
        mock_store.count.return_value = 99
        with patch("src.rag_pipeline.settings") as mock_settings:
            mock_settings.qdrant_collection = "documents"
            mock_settings.embedding_model = "BAAI/bge-m3"
            mock_settings.generator_model = "Qwen/Qwen1.5-4B-Chat"
            result = p.stats()
        assert result["total_chunks"] == 99
