"""Tests for Retriever and format_docs helper."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.retriever import Retriever, format_docs


# ================================================================== format_docs (pure function)

class TestFormatDocs:
    def test_empty_list_returns_no_context_message(self):
        result = format_docs([])
        assert "No relevant context found" in result

    def test_single_doc_contains_source(self):
        docs = [Document(page_content="Test content.", metadata={"source": "test.txt"})]
        result = format_docs(docs)
        assert "[1]" in result
        assert "test.txt" in result
        assert "Test content." in result

    def test_multiple_docs_are_numbered(self):
        docs = [
            Document(page_content="First.", metadata={"source": "a.txt"}),
            Document(page_content="Second.", metadata={"source": "b.txt"}),
            Document(page_content="Third.", metadata={"source": "c.txt"}),
        ]
        result = format_docs(docs)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_docs_are_separated_by_divider(self):
        docs = [
            Document(page_content="Doc A", metadata={"source": "a.txt"}),
            Document(page_content="Doc B", metadata={"source": "b.txt"}),
        ]
        result = format_docs(docs)
        assert "---" in result

    def test_russian_content_preserved(self):
        docs = [
            Document(
                page_content="Qdrant — векторная база данных.",
                metadata={"source": "ru.txt"},
            )
        ]
        result = format_docs(docs)
        assert "Qdrant" in result
        assert "векторная" in result
        assert "ru.txt" in result

    def test_missing_source_metadata_uses_unknown(self):
        docs = [Document(page_content="No source.", metadata={})]
        result = format_docs(docs)
        assert "unknown" in result

    def test_content_order_preserved(self):
        docs = [
            Document(page_content="Alpha", metadata={"source": "a.txt"}),
            Document(page_content="Beta", metadata={"source": "b.txt"}),
        ]
        result = format_docs(docs)
        assert result.index("Alpha") < result.index("Beta")


# ================================================================== Retriever class

class TestRetriever:
    @pytest.fixture
    def mock_lc_retriever(self):
        mock = MagicMock()
        mock.invoke.return_value = [
            Document(page_content="Relevant chunk.", metadata={"source": "doc.txt"}),
        ]
        return mock

    def test_retrieve_calls_lc_retriever_invoke(self, mock_lc_retriever):
        retriever = Retriever(mock_lc_retriever)
        retriever.retrieve("What is Qdrant?")
        mock_lc_retriever.invoke.assert_called_once_with("What is Qdrant?")

    def test_retrieve_returns_documents(self, mock_lc_retriever):
        retriever = Retriever(mock_lc_retriever)
        docs = retriever.retrieve("test query")
        assert len(docs) == 1
        assert docs[0].page_content == "Relevant chunk."

    def test_retrieve_with_context_returns_tuple(self, mock_lc_retriever):
        retriever = Retriever(mock_lc_retriever)
        docs, context = retriever.retrieve_with_context("query")
        assert isinstance(docs, list)
        assert isinstance(context, str)

    def test_retrieve_with_context_formats_correctly(self, mock_lc_retriever):
        retriever = Retriever(mock_lc_retriever)
        docs, context = retriever.retrieve_with_context("What is Qdrant?")
        assert "doc.txt" in context
        assert "Relevant chunk." in context

    def test_retrieve_with_empty_results(self):
        mock_lc = MagicMock()
        mock_lc.invoke.return_value = []
        retriever = Retriever(mock_lc)
        docs, context = retriever.retrieve_with_context("unknown question")
        assert docs == []
        assert "No relevant context found" in context

    def test_retrieve_russian_query(self, mock_lc_retriever):
        mock_lc_retriever.invoke.return_value = [
            Document(page_content="Qdrant — векторная база.", metadata={"source": "ru.txt"})
        ]
        retriever = Retriever(mock_lc_retriever)
        docs, context = retriever.retrieve_with_context("Что такое Qdrant?")
        mock_lc_retriever.invoke.assert_called_once_with("Что такое Qdrant?")
        assert "векторная" in context
