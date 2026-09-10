"""Application service layer for running the Enterprise Knowledge Assistant workflow."""

from typing import Any
from src.config import Settings, get_settings
from src.graph.workflow import build_graph
from src.workflow_middleware import after_workflow, before_workflow


def run_workflow(
    question: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Execute the compiled LangGraph workflow with input and output guardrails.

    Execution Flow:
      1. before_workflow(): input validation and PII sanitization
      2. Multi-agent graph: Retriever Agent -> Response Agent -> Evaluator Agent
      3. after_workflow(): output guardrails verifying grounding, RAGAS quality, and safety
    """
    validated_question = before_workflow(question)

    if settings is None:
        settings = get_settings()

    app = build_graph()
    raw_result = app.invoke({"question": validated_question})

    return after_workflow(raw_result)
