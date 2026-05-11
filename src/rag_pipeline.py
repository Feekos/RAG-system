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

    def query(self, question: str, stream: bool = False) -> RAGResponse:
        if self._gen_chain is None:
            if self._generator is None:
                self._generator = Generator()
            self._build_gen_chain(self._generator)

        docs, context = self._retriever.retrieve_with_context(question)
        if not docs:
            return RAGResponse(
                answer="No relevant documents found in the knowledge base.",
                question=question,
            )

        payload = {"question": question, "context": context}

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

        return RAGResponse(answer=answer, question=question, retrieved_docs=docs)

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict:
        return {
            "collection": settings.qdrant_collection,
            "total_chunks": self._store.count(),
            "embedding_model": settings.embedding_model,
            "generator_model": settings.generator_model,
        }
