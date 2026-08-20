"""Tests for context formatting and source extraction in answering module."""

import unittest
from langchain_core.documents import Document
from src.answering import format_context, get_source_names


class TestQueryFormatting(unittest.TestCase):
    def test_format_context_with_documents(self) -> None:
        docs = [
            Document(
                page_content="PTO rollover is limited to 5 days.",
                metadata={"filename": "leave_policy.md"},
            ),
            Document(
                page_content="Core hours are 10:00 AM to 4:00 PM.",
                metadata={"filename": "remote_work_policy.md", "page": 0},
            ),
        ]

        context = format_context(docs)

        # Check document numbering and filenames in formatted string
        self.assertIn("[1] Source: leave_policy.md", context)
        self.assertIn("PTO rollover is limited to 5 days.", context)
        self.assertIn("[2] Source: remote_work_policy.md (page 1)", context)
        self.assertIn("Core hours are 10:00 AM to 4:00 PM.", context)

    def test_format_context_empty_documents(self) -> None:
        context = format_context([])
        self.assertEqual(context, "No relevant context found.")

    def test_get_source_names_unique_preservation(self) -> None:
        docs = [
            Document(page_content="Chunk 1", metadata={"filename": "leave_policy.md"}),
            Document(page_content="Chunk 2", metadata={"filename": "leave_policy.md"}),
            Document(page_content="Chunk 3", metadata={"filename": "remote_work_policy.md"}),
        ]

        sources = get_source_names(docs)
        self.assertEqual(sources, ["leave_policy.md", "remote_work_policy.md"])


if __name__ == "__main__":
    unittest.main()
