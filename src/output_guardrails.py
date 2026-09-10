"""Output guardrails for validating and sanitizing assistant responses."""

import math
import re
from typing import Any

DEFAULT_GUARDRAIL_MESSAGE = (
    "Unable to show a verified answer for this question. "
    "Please try a more specific policy question."
)

# Minimum quality thresholds for RAGAS evaluation metrics
MIN_FAITHFULNESS_THRESHOLD = 0.70
MIN_ANSWER_RELEVANCY_THRESHOLD = 0.60

# Regex patterns matching real secret-like tokens
SECRET_PATTERNS = [
    # Google AI / Cloud API key pattern (e.g., AIzaSy...)
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),
    # OpenAI API key pattern (e.g., sk-... or sk-proj-...)
    re.compile(r"\bsk-[a-zA-Z0-9_\-]{20,}\b"),
    # LangSmith personal token pattern (e.g., lsv2_pt_...)
    re.compile(r"\blsv2_pt_[a-zA-Z0-9_\-]{10,}\b"),
    # GitHub personal access token (classic ghp_... or fine-grained github_pat_...)
    re.compile(r"\b(?:ghp_[a-zA-Z0-9]{20,}|github_pat_[a-zA-Z0-9_]{20,})\b"),
]

# Patterns detecting prompt-injection instructions or system prompt disclosure attempts
UNSAFE_INSTRUCTION_PATTERNS = [
    re.compile(r"\bignore\s+previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\breveal\s+system\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bshow\s+hidden\s+instructions\b", re.IGNORECASE),
]


class OutputGuardrailError(Exception):
    """Raised when generated response fails post-workflow output safety checks."""

    def __init__(self, message: str = DEFAULT_GUARDRAIL_MESSAGE) -> None:
        super().__init__(message)
        self.message = message


def _is_valid_metric_score(val: Any) -> bool:
    """Validate that score is a finite, non-null numeric float or int."""
    if val is None or isinstance(val, bool):
        return False
    if not isinstance(val, (int, float)):
        try:
            val = float(val)
        except (TypeError, ValueError):
            return False
    return not math.isnan(val) and not math.isinf(val)


def validate_output(
    answer: Any,
    sources: Any,
    evaluation: Any,
    is_redirect: bool = False,
    contact: dict[str, str] | None = None,
) -> None:
    """
    Validate the workflow output to ensure safety, quality, and grounding.

    For normal answers (is_redirect=False):
      - Blocks if answer is missing/empty/whitespace-only.
      - Blocks if source documents are missing or empty.
      - Blocks if RAGAS faithfulness score is missing, invalid, or < 0.70.
      - Blocks if RAGAS answer relevancy score is missing, invalid, or < 0.60.
      - Blocks if answer contains secret-like credentials.
      - Blocks if answer discloses system prompts or contains prompt injections.

    For fixed contact redirects (is_redirect=True):
      - Bypasses source and RAGAS requirements.
      - Validates non-empty answer.
      - Validates presence of configured contact name, position, and email.
      - Blocks if answer contains secret-like credentials or prompt injections.
    """
    # 1. Answer non-empty check
    if not answer or not isinstance(answer, str) or not answer.strip():
        raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

    answer_text = answer.strip()

    # Secret pattern checks
    for pattern in SECRET_PATTERNS:
        if pattern.search(answer_text):
            raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

    # System prompt disclosure / prompt injection checks
    for pattern in UNSAFE_INSTRUCTION_PATTERNS:
        if pattern.search(answer_text):
            raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

    # Contact redirect validation path
    if is_redirect:
        if not contact or not isinstance(contact, dict):
            raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

        c_name = contact.get("name", "").strip()
        c_pos = contact.get("position", "").strip()
        c_email = contact.get("email", "").strip()

        if not (c_name and c_pos and c_email):
            raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

        if c_name not in answer_text or c_pos not in answer_text or c_email not in answer_text:
            raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

        return

    # Normal answer validation path
    # 2. Source documents check
    if not sources or not isinstance(sources, (list, tuple)):
        raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

    valid_sources = [s for s in sources if s and str(s).strip()]
    if not valid_sources:
        raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

    # 3. RAGAS evaluation checks
    if not isinstance(evaluation, dict):
        raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

    faithfulness_raw = evaluation.get("faithfulness")
    if not _is_valid_metric_score(faithfulness_raw):
        raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

    faithfulness = float(faithfulness_raw)
    if faithfulness < MIN_FAITHFULNESS_THRESHOLD:
        raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

    relevancy_raw = evaluation.get("answer_relevancy")
    if not _is_valid_metric_score(relevancy_raw):
        raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)

    relevancy = float(relevancy_raw)
    if relevancy < MIN_ANSWER_RELEVANCY_THRESHOLD:
        raise OutputGuardrailError(DEFAULT_GUARDRAIL_MESSAGE)
