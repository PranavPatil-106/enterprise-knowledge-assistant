"""Graph state definition for LangGraph workflow."""

from typing import Any, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """Represents the complete state passed through the LangGraph workflow."""

    question: str
    documents: list[Document]
    context: str
    mcp_context: str
    answer: str
    sources: list[str]
    evaluation: dict[str, Any] | str
    has_verified_context: bool
    contact: dict[str, str]
    is_redirect: bool
    contact_topic: str
