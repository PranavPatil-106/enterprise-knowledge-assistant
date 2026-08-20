"""Observability and tracing configuration using LangSmith."""

import os
from src.config import Settings, get_settings


def configure_langsmith(settings: Settings | None = None) -> bool:
    """
    Configure LangSmith tracing for LangGraph workflow execution.

    Returns:
        bool: True if tracing is active, False if tracing is disabled.

    Raises:
        ValueError: If tracing is enabled but no LANGSMITH_API_KEY is provided.
    """
    if settings is None:
        settings = get_settings()

    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    if not settings.langsmith_api_key or not settings.langsmith_api_key.strip():
        raise ValueError(
            "LANGSMITH_API_KEY is required when LANGSMITH_TRACING is enabled. "
            "Please provide your API key in the .env file."
        )

    # Configure standard LangSmith and LangChain tracing environment variables
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.strip()
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.strip()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project.strip()
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project.strip()

    endpoint = settings.langsmith_endpoint.strip() if settings.langsmith_endpoint else "https://api.smith.langchain.com"
    os.environ["LANGSMITH_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    return True


def flush_langsmith() -> None:
    """Safely flush pending LangSmith traces before process termination."""
    try:
        from langsmith import Client

        client = Client()
        if hasattr(client, "flush"):
            client.flush()
    except Exception:
        pass
