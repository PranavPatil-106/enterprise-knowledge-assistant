"""Application-level workflow middleware for safety validation before and after graph execution."""

from typing import Any
from src.guardrails import validate_question
from src.output_guardrails import OutputGuardrailError, validate_output


def before_workflow(question: str) -> str:
    """
    Validate user input before invoking the LangGraph workflow.

    Applies input guardrails to prevent empty questions, oversized prompts,
    and sanitize PII.
    """
    return validate_question(question)


def after_workflow(result: dict[str, Any]) -> dict[str, Any]:
    """
    Validate workflow outputs after graph execution before returning to the UI or caller.

    Preserves foundational answer and source existence checks for verified answers,
    supports safe contact redirection validation when no verified policy context was retrieved,
    and runs comprehensive output guardrails.

    Pipeline execution order:
    Retriever Agent -> Response Agent -> Evaluator Agent -> Output Guardrails -> Final Result
    """
    if not isinstance(result, dict):
        raise OutputGuardrailError("Workflow execution did not return a valid result state.")

    answer = result.get("answer")
    if not answer or not str(answer).strip():
        raise OutputGuardrailError(
            "Unable to show a verified answer for this question. Please try a more specific policy question."
        )

    sources = result.get("sources", [])
    is_redirect = bool(result.get("is_redirect", False))
    contact = result.get("contact")

    if not is_redirect:
        if not sources or len(sources) == 0:
            raise OutputGuardrailError(
                "Unable to show a verified answer for this question. Please try a more specific policy question."
            )

    evaluation = result.get("evaluation")

    validate_output(
        answer=answer,
        sources=sources,
        evaluation=evaluation,
        is_redirect=is_redirect,
        contact=contact,
    )

    return result
