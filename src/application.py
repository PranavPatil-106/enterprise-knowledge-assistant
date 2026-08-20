"""Application service layer for running the Enterprise Knowledge Assistant workflow."""

from typing import Any
from src.config import Settings, get_settings
from src.graph.workflow import build_graph
from src.observability import configure_langsmith, flush_langsmith
from src.workflow_middleware import after_workflow, before_workflow


def run_workflow(
    question: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Execute the compiled LangGraph workflow for a question with guardrails and LangSmith observability.

    1. Validates input via before_workflow() guardrails.
    2. Configures LangSmith tracing before execution.
    3. Runs the multi-agent graph (Retriever Agent -> Response Agent -> Evaluator Agent).
    4. Validates output via after_workflow() (ensuring answer and sources).
    5. Ensures pending traces are flushed upon completion.

    Returns the resulting GraphState dictionary.
    """
    validated_question = before_workflow(question)

    if settings is None:
        settings = get_settings()

    tracing_enabled = configure_langsmith(settings)

    try:
        app = build_graph()
        final_state = app.invoke({"question": validated_question})
        return after_workflow(final_state)
    finally:
        if tracing_enabled:
            flush_langsmith()
