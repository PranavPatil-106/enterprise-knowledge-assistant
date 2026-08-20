"""Tests for document chunking logic."""

import unittest
from langchain_core.documents import Document
from src.rag.indexer import create_chunks


class TestDocumentChunking(unittest.TestCase):
    def test_chunking_splits_large_text(self) -> None:
        # Sample in-memory text with several paragraphs
        sample_text = (
            "Section 1: Introduction\n"
            "This is a test paragraph for the enterprise knowledge base. " * 15
            + "\n\n"
            + "Section 2: Details\n"
            + "Here are additional details regarding enterprise policy rules. " * 15
        )

        sample_doc = Document(
            page_content=sample_text,
            metadata={"source": "memory_test.md", "filename": "memory_test.md"},
        )

        chunks = create_chunks([sample_doc], chunk_size=300, chunk_overlap=50)

        # Verify chunks were created
        self.assertGreater(len(chunks), 1, "Long text should be split into multiple chunks")

        # Verify all chunks maintain metadata
        for chunk in chunks:
            self.assertEqual(chunk.metadata["filename"], "memory_test.md")
            self.assertLessEqual(len(chunk.page_content), 350)

    def test_chunking_short_text_single_chunk(self) -> None:
        short_doc = Document(
            page_content="Short policy summary text.",
            metadata={"source": "short.txt", "filename": "short.txt"},
        )

        chunks = create_chunks([short_doc], chunk_size=800, chunk_overlap=120)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, "Short policy summary text.")


if __name__ == "__main__":
    unittest.main()
