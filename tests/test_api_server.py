from __future__ import annotations

import json
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest

from src import api_server


class TestApiPathResolution:
    def test_resolve_data_path_allows_data_documents(self):
        resolved = api_server._resolve_data_path("data/documents")
        assert resolved.name == "documents"

    def test_resolve_data_path_rejects_outside_data(self):
        with pytest.raises(ValueError, match="Only paths inside"):
            api_server._resolve_data_path("README.md")


class TestApiHandler:
    def test_health_returns_ok_without_loading_pipeline(self):
        handler = api_server.RAGRequestHandler.__new__(api_server.RAGRequestHandler)
        handler.path = "/health"
        handler.request_version = "HTTP/1.1"
        handler.command = "GET"
        handler.responses = []

        body = {}

        def write_json(payload, status=HTTPStatus.OK):
            body["payload"] = payload
            body["status"] = status

        handler._write_json = write_json

        with patch("src.api_server._get_pipeline") as mock_get_pipeline:
            handler.do_GET()

        mock_get_pipeline.assert_not_called()
        assert body["payload"] == {"status": "ok"}
        assert body["status"] == HTTPStatus.OK

    def test_query_requires_non_empty_question(self):
        handler = api_server.RAGRequestHandler.__new__(api_server.RAGRequestHandler)
        with pytest.raises(ValueError, match="question"):
            handler._handle_query({"question": "   "})

    def test_query_returns_answer_and_sources(self):
        handler = api_server.RAGRequestHandler.__new__(api_server.RAGRequestHandler)
        body = {}
        handler._write_json = lambda payload, status=HTTPStatus.OK: body.update(
            {"payload": payload, "status": status}
        )

        response = MagicMock()
        response.question = "What is RAG?"
        response.answer = "RAG combines retrieval and generation."
        response.sources = ["sample.txt"]

        pipeline = MagicMock()
        pipeline.query.return_value = response

        with patch("src.api_server._get_pipeline", return_value=pipeline):
            handler._handle_query({"question": "What is RAG?"})

        assert body["payload"] == {
            "question": "What is RAG?",
            "answer": "RAG combines retrieval and generation.",
            "sources": ["sample.txt"],
        }
        assert json.dumps(body["payload"])
