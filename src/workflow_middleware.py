"""Application-level workflow middleware for safety validation before and after graph execution."""

from typing import Any
from src.guardrails import validate_question


def before_workflow(question: str) -> str:
    """
    Validate user input before invoking the LangGraph workflow.

    Applies guardrails to prevent empty questions, oversized prompts,
    prompt injections, and requests for secret keys.
    """
    return validate_question(question)


def after_workflow(result: dict[str, Any]) -> dict[str, Any]:
    """
    Validate workflow outputs after graph execution before returning to the UI or caller.

    Ensures:
      - A non-empty grounded answer was produced.
      - At least one authoritative source document was retrieved and cited.

    Raises ValueError with a user-friendly error message if verification fails.
    """
    if not isinstance(result, dict):
        raise ValueError("Workflow execution did not return a valid result state.")

    answer = result.get("answer")
    if not answer or not str(answer).strip():
        raise ValueError(
            "The knowledge assistant was unable to generate an answer. "
            "No valid response was produced by the response agent."
        )

    sources = result.get("sources", [])
    if not sources or len(sources) == 0:
        raise ValueError(
            "No authoritative source documents were retrieved to ground this answer. "
            "Unsubstantiated answers are prohibited."
        )

    return result
