"""Tests for LangSmith observability and tracing configuration."""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.observability import configure_langsmith, flush_langsmith


class TestObservability(unittest.TestCase):
    def setUp(self) -> None:
        # Save existing env vars
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        # Restore environment
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_tracing_disabled_does_not_require_key(self) -> None:
        """When LANGSMITH_TRACING=false, configure_langsmith returns False without requiring a key."""
        mock_settings = MagicMock()
        mock_settings.langsmith_tracing = False
        mock_settings.langsmith_api_key = ""
        mock_settings.langsmith_project = "enterprise-knowledge-assistant"
        mock_settings.langsmith_endpoint = "https://api.smith.langchain.com"

        result = configure_langsmith(mock_settings)

        self.assertFalse(result)
        self.assertEqual(os.environ.get("LANGSMITH_TRACING"), "false")
        self.assertEqual(os.environ.get("LANGCHAIN_TRACING_V2"), "false")

    def test_tracing_enabled_with_key_applies_env(self) -> None:
        """When LANGSMITH_TRACING=true with a key, environment variables are configured and returns True."""
        mock_settings = MagicMock()
        mock_settings.langsmith_tracing = True
        mock_settings.langsmith_api_key = "lsv2_pt_dummy_test_key_12345"
        mock_settings.langsmith_project = "test-project-name"
        mock_settings.langsmith_endpoint = "https://api.smith.langchain.com"

        result = configure_langsmith(mock_settings)

        self.assertTrue(result)
        self.assertEqual(os.environ.get("LANGSMITH_TRACING"), "true")
        self.assertEqual(os.environ.get("LANGCHAIN_TRACING_V2"), "true")
        self.assertEqual(os.environ.get("LANGSMITH_API_KEY"), "lsv2_pt_dummy_test_key_12345")
        self.assertEqual(os.environ.get("LANGCHAIN_API_KEY"), "lsv2_pt_dummy_test_key_12345")
        self.assertEqual(os.environ.get("LANGSMITH_PROJECT"), "test-project-name")
        self.assertEqual(os.environ.get("LANGCHAIN_PROJECT"), "test-project-name")
        self.assertEqual(os.environ.get("LANGSMITH_ENDPOINT"), "https://api.smith.langchain.com")

    def test_tracing_enabled_without_key_raises_error(self) -> None:
        """When LANGSMITH_TRACING=true but no API key is provided, raises ValueError."""
        mock_settings = MagicMock()
        mock_settings.langsmith_tracing = True
        mock_settings.langsmith_api_key = ""
        mock_settings.langsmith_project = "enterprise-knowledge-assistant"
        mock_settings.langsmith_endpoint = "https://api.smith.langchain.com"

        with self.assertRaises(ValueError) as ctx:
            configure_langsmith(mock_settings)

        self.assertIn("LANGSMITH_API_KEY is required", str(ctx.exception))

    @patch("langsmith.Client")
    def test_flush_langsmith_invokes_client_flush(self, mock_client_cls) -> None:
        """flush_langsmith safely invokes client.flush() if client exists."""
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        flush_langsmith()
        mock_instance.flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()
