from typing import Any, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str
    documents: list[Document]
    context: str
    mcp_context: str
    answer: str
    sources: list[str]
    evaluation: dict[str, Any] | str
    retry_count: int
    next_step: str
