from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

_MULTILINGUAL_SEPARATORS = [
    "\n\n",   # paragraph break
    "\n",     # line break
    ". ",     # sentence end (EN / RU)
    "! ",     # exclamation
    "? ",     # question
    ".\n",    # sentence end before newline
    "!\n",
    "?\n",
    "; ",     # clause separator
    ", ",     # comma clause
    " ",      # word break
    "",       # character fallback
]

class DocumentProcessor:
    """
    Loads .txt / .md / .rst / .pdf files and splits them into LangChain Documents.
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            separators=_MULTILINGUAL_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    def load_file(self, path: str | Path) -> List[Document]:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in (".txt", ".md", ".rst"):
            loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        elif suffix == ".pdf":
            try:
                loader = PyPDFLoader(str(path))
            except ImportError:
                raise ImportError("Install pypdf for PDF support: pip install pypdf")
        else:
            raise ValueError(
                f"Неподдерживаемый тип файла: {suffix!r}. Поддерживаемые форматы: .txt .md .rst .pdf"
            )

        raw_docs = loader.load()
        # Normalise source metadata to filename only (not the full path)
        for doc in raw_docs:
            doc.metadata["source"] = path.name

        return self._splitter.split_documents(raw_docs)

    def load_directory(self, directory: str | Path) -> List[Document]:
        directory = Path(directory)
        all_chunks: List[Document] = []
        supported = {".txt", ".md", ".rst", ".pdf"}

        for file_path in sorted(directory.rglob("*")):
            if file_path.suffix.lower() in supported:
                print(f"[Processor] Загрузка {file_path.name} ...")
                try:
                    all_chunks.extend(self.load_file(file_path))
                except Exception as exc:
                    print(f"[Processor] Пропущено {file_path.name}: {exc}")

        return all_chunks
