from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


def format_docs(docs: List[Document]) -> str:
    """Format a list of retrieved Documents into a numbered context string."""
    if not docs:
        return "No relevant context found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class Retriever:
    """Thin wrapper around a LangChain VectorStoreRetriever."""

    def __init__(self, lc_retriever: VectorStoreRetriever):
        self._lc = lc_retriever

    def retrieve(self, query: str) -> List[Document]:
        return self._lc.invoke(query)

    def retrieve_with_context(self, query: str) -> Tuple[List[Document], str]:
        docs = self.retrieve(query)
        return docs, format_docs(docs)
