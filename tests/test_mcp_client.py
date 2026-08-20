"""Tests for the Filesystem MCP client and path validation."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.mcp_client import (
    extract_text_from_mcp_result,
    read_text_file_via_mcp,
    validate_file_path,
)


class TestMcpClient(unittest.TestCase):
    def setUp(self) -> None:
        self.allowed_dir = Path("data/raw").resolve()
        self.allowed_dir.mkdir(parents=True, exist_ok=True)
        self.valid_file = self.allowed_dir / "leave_policy.md"

    def test_validate_file_path_success(self) -> None:
        """Valid file inside allowed directory resolves safely."""
        validated = validate_file_path(self.valid_file, self.allowed_dir)
        self.assertEqual(validated, self.valid_file.resolve())

    def test_validate_file_path_outside_directory(self) -> None:
        """Paths outside allowed directory must raise ValueError."""
        outside_file = Path("pyproject.toml").resolve()
        with self.assertRaises(ValueError) as ctx:
            validate_file_path(outside_file, self.allowed_dir)
        self.assertIn("Access denied", str(ctx.exception))

    def test_validate_file_path_nonexistent(self) -> None:
        """Non-existent files inside allowed directory must raise FileNotFoundError."""
        nonexistent = self.allowed_dir / "does_not_exist.md"
        with self.assertRaises(FileNotFoundError):
            validate_file_path(nonexistent, self.allowed_dir)

    def test_extract_text_from_mcp_result_success(self) -> None:
        """TextContent objects should be extracted and joined properly."""
        mock_item1 = MagicMock()
        mock_item1.text = "# Policy Title"
        mock_item2 = MagicMock()
        mock_item2.text = "Policy details paragraph."

        mock_result = MagicMock()
        mock_result.is_error = False
        mock_result.content = [mock_item1, mock_item2]

        text = extract_text_from_mcp_result(mock_result)
        self.assertIn("# Policy Title", text)
        self.assertIn("Policy details paragraph.", text)

    def test_extract_text_from_mcp_result_error(self) -> None:
        """Results with is_error=True must raise RuntimeError."""
        mock_result = MagicMock()
        mock_result.is_error = True
        with self.assertRaises(RuntimeError):
            extract_text_from_mcp_result(mock_result)

    def test_extract_text_from_mcp_result_empty(self) -> None:
        """Results with empty content must raise ValueError."""
        mock_result = MagicMock()
        mock_result.is_error = False
        mock_result.content = []
        with self.assertRaises(ValueError):
            extract_text_from_mcp_result(mock_result)

    @patch("src.mcp_client._async_read_text_file")
    def test_read_text_file_via_mcp_mocked(self, mock_async_read) -> None:
        """Synchronous wrapper correctly delegates to async client and returns text."""
        mock_async_read.return_value = "Mocked MCP document text content."

        content = read_text_file_via_mcp(self.valid_file, self.allowed_dir)
        self.assertEqual(content, "Mocked MCP document text content.")
        mock_async_read.assert_called_once()


if __name__ == "__main__":
    unittest.main()
