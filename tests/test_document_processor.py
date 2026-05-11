"""Tests for DocumentProcessor — chunking and file loading logic."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.document_processor import DocumentProcessor, _MULTILINGUAL_SEPARATORS


# ================================================================== splitter logic

class TestChunking:
    def test_english_text_is_split_into_multiple_chunks(self, tmp_english_file):
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        chunks = processor.load_file(tmp_english_file)
        # Text is > 200 chars so must produce more than one chunk
        assert len(chunks) > 1

    def test_russian_text_is_split_into_multiple_chunks(self, tmp_russian_file):
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        chunks = processor.load_file(tmp_russian_file)
        assert len(chunks) > 1

    def test_chunk_size_not_exceeded(self, tmp_english_file):
        chunk_size = 300
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=30)
        chunks = processor.load_file(tmp_english_file)
        # Allow a small margin — splitter tries its best but can't always split at exact boundary
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 50, (
                f"Chunk too large ({len(chunk.page_content)} chars): {chunk.page_content[:80]!r}"
            )

    def test_russian_chunk_size_not_exceeded(self, tmp_russian_file):
        chunk_size = 250
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=25)
        chunks = processor.load_file(tmp_russian_file)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 50

    def test_cyrillic_characters_preserved(self, tmp_russian_file):
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        chunks = processor.load_file(tmp_russian_file)
        all_text = " ".join(c.page_content for c in chunks)
        assert "Qdrant" in all_text
        assert "векторная" in all_text
        assert "языковые" in all_text or "языков" in all_text

    def test_no_content_lost_in_chunking(self, tmp_path):
        """
        All words from the source text must appear in at least one chunk.
        This is the most critical RAG property — silent content loss breaks retrieval.
        """
        words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
        text = ". ".join(words) + "."
        f = tmp_path / "words.txt"
        f.write_text(text, encoding="utf-8")

        processor = DocumentProcessor(chunk_size=30, chunk_overlap=5)
        chunks = processor.load_file(f)
        combined = " ".join(c.page_content for c in chunks)

        for word in words:
            assert word in combined, f"Word '{word}' was lost during chunking"

    def test_mixed_language_text_preserved(self, tmp_mixed_file):
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=30)
        chunks = processor.load_file(tmp_mixed_file)
        all_text = " ".join(c.page_content for c in chunks)
        assert "Qdrant" in all_text
        assert "векторная" in all_text

    def test_small_text_returns_single_chunk(self, tmp_path):
        small = tmp_path / "small.txt"
        small.write_text("Short text.", encoding="utf-8")
        processor = DocumentProcessor(chunk_size=512, chunk_overlap=64)
        chunks = processor.load_file(small)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_empty_file_returns_no_chunks(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        processor = DocumentProcessor()
        chunks = processor.load_file(empty)
        assert chunks == []


# ================================================================== metadata

class TestMetadata:
    def test_source_metadata_is_filename_only(self, tmp_english_file):
        processor = DocumentProcessor()
        chunks = processor.load_file(tmp_english_file)
        for chunk in chunks:
            assert chunk.metadata["source"] == "english.txt", (
                f"Expected filename only, got: {chunk.metadata['source']!r}"
            )

    def test_russian_source_metadata_is_filename_only(self, tmp_russian_file):
        processor = DocumentProcessor()
        chunks = processor.load_file(tmp_russian_file)
        for chunk in chunks:
            assert chunk.metadata["source"] == "russian.txt"

    def test_all_chunks_have_source_key(self, tmp_english_file):
        processor = DocumentProcessor()
        chunks = processor.load_file(tmp_english_file)
        for chunk in chunks:
            assert "source" in chunk.metadata


# ================================================================== file loading

class TestFileLoading:
    def test_txt_file_loads(self, tmp_english_file):
        processor = DocumentProcessor()
        chunks = processor.load_file(tmp_english_file)
        assert len(chunks) >= 1
        assert all(isinstance(c, Document) for c in chunks)

    def test_md_file_loads(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nSome content here.\n", encoding="utf-8")
        processor = DocumentProcessor()
        chunks = processor.load_file(md)
        assert len(chunks) >= 1

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        bad = tmp_path / "file.xyz"
        bad.write_text("content", encoding="utf-8")
        processor = DocumentProcessor()
        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.load_file(bad)

    def test_load_directory_finds_multiple_files(self, tmp_path):
        for name, content in [("a.txt", "File A content " * 20), ("b.txt", "File B content " * 20)]:
            (tmp_path / name).write_text(content, encoding="utf-8")
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        chunks = processor.load_directory(tmp_path)
        sources = {c.metadata["source"] for c in chunks}
        assert "a.txt" in sources
        assert "b.txt" in sources

    def test_load_directory_skips_unsupported_files(self, tmp_path):
        (tmp_path / "valid.txt").write_text("Valid content " * 20, encoding="utf-8")
        (tmp_path / "ignore.csv").write_text("col1,col2\n1,2\n", encoding="utf-8")
        processor = DocumentProcessor()
        chunks = processor.load_directory(tmp_path)
        sources = {c.metadata["source"] for c in chunks}
        assert "ignore.csv" not in sources

    def test_load_directory_with_russian_files(self, tmp_path):
        (tmp_path / "ru.txt").write_text("Это русский текст. " * 30, encoding="utf-8")
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        chunks = processor.load_directory(tmp_path)
        all_text = " ".join(c.page_content for c in chunks)
        assert "русский" in all_text


# ================================================================== separators

class TestSeparators:
    def test_multilingual_separators_contains_paragraph_break(self):
        assert "\n\n" in _MULTILINGUAL_SEPARATORS

    def test_multilingual_separators_contains_sentence_end(self):
        assert ". " in _MULTILINGUAL_SEPARATORS

    def test_multilingual_separators_contains_character_fallback(self):
        # Last separator must be "" to allow character-level splitting as last resort
        assert "" in _MULTILINGUAL_SEPARATORS
