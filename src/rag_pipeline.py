from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from .config import settings
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingModel
from .generator import Generator, build_rag_prompt
from .retriever import Retriever, format_docs
from .vector_store import VectorStore


@dataclass
class RAGResponse:
    answer: str
    question: str
    retrieved_docs: List[Document] = field(default_factory=list)

    @property
    def sources(self) -> List[str]:
        return list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in self.retrieved_docs))


class RAGPipeline:
    """
    LangChain-based RAG pipeline.

    Ingest:  file/dir -> LangChain Documents -> chunk -> embed -> Qdrant
    Query:   question -> retrieve (Qdrant ANN) -> format context -> LCEL chain -> answer
    """

    def __init__(
        self,
        processor: DocumentProcessor,
        store: VectorStore,
        retriever: Retriever,
        generator: Generator | None = None,
    ):
        self._processor = processor
        self._store = store
        self._retriever = retriever
        self._generator = generator
        self._gen_chain: RunnableSerializable | None = None
        self._chat_histories: dict[str, list[tuple[str, str]]] = {}
        if generator:
            self._build_gen_chain(generator)

    @classmethod
    def create(cls, lazy_generator: bool = False, top_k: int | None = None) -> "RAGPipeline":
        """Factory that wires all components together."""
        processor = DocumentProcessor()
        embeddings = EmbeddingModel()
        store = VectorStore(embeddings)
        retriever = Retriever(store.as_retriever(top_k=top_k))
        generator = None if lazy_generator else Generator()
        return cls(processor, store, retriever, generator)

    def _build_gen_chain(self, generator: Generator) -> None:
        # LCEL chain: prompt | llm | output parser
        self._gen_chain = build_rag_prompt() | generator.langchain | StrOutputParser()

    # ------------------------------------------------------------------ ingest

    def ingest_file(self, path: str, batch_size: int = 64) -> int:
        return self._index_docs(self._processor.load_file(path), batch_size)

    def ingest_directory(self, directory: str, batch_size: int = 64) -> int:
        return self._index_docs(self._processor.load_directory(directory), batch_size)

    def _index_docs(self, docs: List[Document], batch_size: int) -> int:
        if not docs:
            print("[Pipeline] No documents to index.")
            return 0

        total = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self._store.add_documents(batch)
            total += len(batch)
            print(f"[Pipeline] Indexed {total}/{len(docs)} chunks ...")

        print(f"[Pipeline] Done. Total: {total} chunks.")
        return total

    def reset_collection(self) -> None:
        self._store.delete_collection()

    # ------------------------------------------------------------------ query

    def query(self, question: str, stream: bool = False, session_id: str = "default", use_history: bool = True) -> RAGResponse:
        if self._gen_chain is None:
            if self._generator is None:
                self._generator = Generator()
            self._build_gen_chain(self._generator)

        history = self._get_history(session_id) if use_history else []
        retrieval_query = self._build_retrieval_query(question, history)
        docs, context = self._retriever.retrieve_with_context(retrieval_query)
        if not docs:
            return RAGResponse(
                answer="No relevant documents found in the knowledge base.",
                question=question,
            )

        payload = {
            "question": question,
            "context": context,
            "chat_history": self._format_chat_history(history),
        }

        if stream:
            print("Assistant: ", end="", flush=True)
            parts: List[str] = []
            for chunk in self._gen_chain.stream(payload):
                print(chunk, end="", flush=True)
                parts.append(chunk)
            print()
            answer = "".join(parts)
        else:
            answer = self._gen_chain.invoke(payload)

        if use_history:
            self._remember_turn(session_id, question, answer)

        return RAGResponse(answer=answer, question=question, retrieved_docs=docs)

    def clear_history(self, session_id: str = "default") -> None:
        self._chat_histories.pop(session_id, None)

    def _get_history(self, session_id: str) -> list[tuple[str, str]]:
        turns = max(0, int(getattr(settings, "context_window_turns", 0)))
        if turns == 0:
            return []
        return list(self._chat_histories.get(session_id, [])[-turns:])

    def _remember_turn(self, session_id: str, question: str, answer: str) -> None:
        turns = max(0, int(getattr(settings, "context_window_turns", 0)))
        if turns == 0:
            return
        history = self._chat_histories.setdefault(session_id, [])
        history.append((question, answer))
        if len(history) > turns:
            del history[:-turns]

    @staticmethod
    def _format_chat_history(history: list[tuple[str, str]]) -> str:
        if not history:
            return "No prior conversation."
        parts = []
        for index, (question, answer) in enumerate(history, 1):
            parts.append(f"[{index}] User: {question}\n[{index}] Assistant: {answer}")
        return "\n\n".join(parts)

    @classmethod
    def _build_retrieval_query(cls, question: str, history: list[tuple[str, str]]) -> str:
        if not history:
            return question
        return (
            "Conversation history:\n"
            f"{cls._format_chat_history(history)}\n\n"
            f"Current question: {question}"
        )

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict:
        return {
            "collection": settings.qdrant_collection,
            "total_chunks": self._store.count(),
            "embedding_model": settings.embedding_model,
            "generator_model": settings.generator_model,
            "context_window_turns": settings.context_window_turns,
        }
