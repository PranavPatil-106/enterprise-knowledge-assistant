from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware.pii import PIIDetectionError

MAX_QUESTION_LENGTH = 500

DEFAULT_PII_MIDDLEWARES = [
    PIIMiddleware("email", strategy="redact"),
    PIIMiddleware("credit_card", strategy="mask"),
    PIIMiddleware("url", strategy="redact"),
    PIIMiddleware("ip", strategy="hash"),
    PIIMiddleware("ssn", detector=r"\b\d{3}-\d{2}-\d{4}\b", strategy="mask"),
]


def apply_pii_middlewares(text: str, middlewares: list[PIIMiddleware] | None = None) -> str:
    active_middlewares = middlewares if middlewares is not None else DEFAULT_PII_MIDDLEWARES
    sanitized = text
    for mw in active_middlewares:
        sanitized, _ = mw._process_content(sanitized)
    return sanitized


def validate_question(question: str, middlewares: list[PIIMiddleware] | None = None) -> str:
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

    try:
        return apply_pii_middlewares(cleaned, middlewares=middlewares)
    except PIIDetectionError as e:
        raise ValueError(f"PII violation detected: {e}") from e
