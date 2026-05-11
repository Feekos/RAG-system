from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

from .rag_pipeline import RAGPipeline


_pipeline: RAGPipeline | None = None
_pipeline_lock = Lock()
_data_root = Path("data").resolve()


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = RAGPipeline.create(lazy_generator=True)
    return _pipeline


def _reset_pipeline() -> None:
    global _pipeline
    with _pipeline_lock:
        _pipeline = None


def _resolve_data_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(_data_root)
    except ValueError as exc:
        raise ValueError(f"Only paths inside '{_data_root}' can be ingested.") from exc
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return resolved


class RAGRequestHandler(BaseHTTPRequestHandler):
    server_version = "RAGHTTP/1.0"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json({"status": "ok"})
            return
        if self.path == "/stats":
            self._write_json(_get_pipeline().stats())
            return
        self._write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/ingest":
                self._handle_ingest(payload)
                return
            if self.path == "/query":
                self._handle_query(payload)
                return
            self._write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as exc:
            self._write_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._write_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[API] {self.address_string()} - {format % args}")

    def _handle_ingest(self, payload: dict[str, Any]) -> None:
        path = str(payload.get("path", "data/documents"))
        reset = bool(payload.get("reset", False))
        target = _resolve_data_path(path)

        if reset:
            pipeline = _get_pipeline()
            pipeline.reset_collection()
            _reset_pipeline()

        pipeline = _get_pipeline()
        count = (
            pipeline.ingest_directory(str(target))
            if target.is_dir()
            else pipeline.ingest_file(str(target))
        )
        self._write_json({"indexed_chunks": count, "stats": pipeline.stats()})

    def _handle_query(self, payload: dict[str, Any]) -> None:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError("Field 'question' must be a non-empty string.")

        response = _get_pipeline().query(question, stream=False)
        self._write_json(
            {
                "question": response.question,
                "answer": response.answer,
                "sources": response.sources,
            }
        )

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(data, dict):
            raise ValueError("Request JSON must be an object.")
        return data

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    host = os.getenv("RAG_INTERNAL_HOST", "0.0.0.0")
    port = int(os.getenv("RAG_INTERNAL_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), RAGRequestHandler)
    print(f"[API] Serving on {host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
