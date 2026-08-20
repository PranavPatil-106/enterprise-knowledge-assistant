"""Input validation and guardrails for the Enterprise Knowledge Assistant."""

import re

MAX_QUESTION_LENGTH = 500

# Patterns indicating prompt-injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?",
    r"reveal\s+(?:the\s+)?(?:system\s+prompt|hidden\s+instructions?)",
    r"show\s+(?:the\s+)?(?:system\s+prompt|hidden\s+instructions?)",
    r"print\s+(?:the\s+)?(?:system\s+prompt|hidden\s+instructions?)",
    r"display\s+(?:the\s+)?(?:system\s+prompt|hidden\s+instructions?)",
    r"output\s+(?:the\s+)?(?:system\s+prompt|hidden\s+instructions?)",
    r"bypass\s+(?:safety|guardrails|instructions?)",
    r"\bjailbreak\b",
]

# Patterns attempting to exfiltrate API keys, credentials, or environment secrets
SECRET_PATTERNS = [
    r"\b(?:api[_\s-]?key|api[_\s-]?keys)\b",
    r"\bgoogle[_\s-]?api[_\s-]?key\b",
    r"\bopenai[_\s-]?api[_\s-]?key\b",
    r"\blangsmith[_\s-]?api[_\s-]?key\b",
    r"\b(?:secret[_\s-]?key|access[_\s-]?token|auth[_\s-]?token|private[_\s-]?key)\b",
    r"\.env(?:\s+file|\s+contents)?\b",
    r"\benvironment\s+variables?\b",
    r"\b(?:show|reveal|give|print|display|tell|expose|leak)\s+(?:me\s+)?(?:the\s+)?(?:actual\s+)?(?:password\s+hash|passwords|credentials|secrets?)\b",
]


def validate_question(question: str) -> str:
    """
    Validate and sanitize user input questions before workflow execution.

    Rejects:
      - Empty or whitespace-only questions
      - Questions exceeding 500 characters
      - Prompt-injection attempts
      - Requests for system secrets, API keys, credentials, or environment variables

    Allows:
      - Legitimate enterprise policy inquiries (e.g. password requirements, leave, remote work)

    Returns the stripped question if valid, or raises ValueError with a user-friendly message.
    """
    if not isinstance(question, str):
        raise ValueError("Question must be a valid text string.")

    cleaned = question.strip()

    if not cleaned:
        raise ValueError("Question cannot be empty. Please enter a valid inquiry.")

    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Question exceeds the maximum allowed length of {MAX_QUESTION_LENGTH} characters "
            f"(received {len(cleaned)} characters)."
        )

    # Check for prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise ValueError(
                "Invalid query: Prompt injection or system instruction manipulation attempt detected."
            )

    # Check for secret / API key exfiltration
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise ValueError(
                "Invalid query: Requests for system secrets, API keys, credentials, or environment variables are strictly prohibited."
            )

    return cleaned
