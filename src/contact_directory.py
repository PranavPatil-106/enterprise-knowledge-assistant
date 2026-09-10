"""Contact directory loading, topic classification, and message formatting helpers."""

import json
import re
from pathlib import Path
from typing import Any

from src.config import get_settings

DEFAULT_FALLBACK_CONTACT: dict[str, str] = {
    "topic_name": "general enterprise support",
    "name": "Enterprise Support Desk",
    "position": "Support Coordinator",
    "email": "support@example.com",
}

DEFAULT_CONTACT_DIRECTORY: dict[str, dict[str, str]] = {
    "it_security": {
        "topic_name": "IT and security",
        "name": "IT Service Desk",
        "position": "IT Support Lead",
        "email": "it-support@example.com",
    },
    "people_operations": {
        "topic_name": "leave and remote work",
        "name": "People Operations Team",
        "position": "People Operations Lead",
        "email": "people-ops@example.com",
    },
    "travel_expense": {
        "topic_name": "business travel and expenses",
        "name": "Finance Operations Desk",
        "position": "Travel & Expense Lead",
        "email": "finance-travel@example.com",
    },
    "ethics_compliance": {
        "topic_name": "ethics and conduct",
        "name": "Ethics & Compliance Office",
        "position": "Compliance Officer",
        "email": "ethics-compliance@example.com",
    },
    "employee_benefits": {
        "topic_name": "employee benefits and insurance",
        "name": "Benefits & Rewards Team",
        "position": "Total Rewards Specialist",
        "email": "benefits@example.com",
    },
    "workplace_safety": {
        "topic_name": "workplace safety and facilities",
        "name": "Facilities & Safety Team",
        "position": "EHS Coordinator",
        "email": "facilities-safety@example.com",
    },
    "general_support": {
        "topic_name": "general policy",
        "name": "Employee Support Desk",
        "position": "Support Coordinator",
        "email": "employee-support@example.com",
    },
}

# Keywords for deterministic topic classification
IT_SECURITY_KEYWORDS = [
    "password",
    "mfa",
    "security",
    "device",
    "laptop",
    "access",
    "encryption",
    "vpn",
]

PEOPLE_OPERATIONS_KEYWORDS = [
    "leave",
    "pto",
    "sick leave",
    "remote work",
    "work from home",
    "holiday",
    "parental leave",
    "bereavement",
]

TRAVEL_EXPENSE_KEYWORDS = [
    "travel",
    "expense",
    "per diem",
    "flight",
    "hotel",
    "lodging",
    "reimbursement",
    "mileage",
    "receipt",
    "airfare",
]

ETHICS_COMPLIANCE_KEYWORDS = [
    "conduct",
    "ethics",
    "harassment",
    "conflict of interest",
    "whistleblower",
    "bribery",
    "gift",
    "integrity",
    "discrimination",
    "retaliation",
]

EMPLOYEE_BENEFITS_KEYWORDS = [
    "benefit",
    "benefits",
    "health insurance",
    "mediclaim",
    "medical plan",
    "dental",
    "vision",
    "epf",
    "provident fund",
    "vpf",
    "nps",
    "gratuity",
    "pension",
    "retirement",
    "wellness stipend",
    "tuition",
    "disability insurance",
    "life insurance",
]

WORKPLACE_SAFETY_KEYWORDS = [
    "safety",
    "evacuation",
    "fire drill",
    "first aid",
    "aed",
    "ergonomic",
    "hazard",
    "injury",
    "visitor badge",
    "emergency exit",
    "warden",
]


def create_default_contact(topic: str | None = None) -> dict[str, str]:
    """
    Create a default contact that users can email when no contact is found in the list.
    """
    topic_clean = topic.replace("_", " ").strip() if topic else "enterprise support"
    if not topic_clean.endswith("support"):
        topic_name = f"{topic_clean} support"
    else:
        topic_name = topic_clean

    return {
        "topic_name": topic_name,
        "name": DEFAULT_FALLBACK_CONTACT["name"],
        "position": DEFAULT_FALLBACK_CONTACT["position"],
        "email": DEFAULT_FALLBACK_CONTACT["email"],
    }


def ensure_contact_directory(path: Path | str | None = None) -> Path:
    """Ensure contact directory JSON file exists on disk, creating default if absent."""
    target_path = Path(path) if path else get_settings().contact_directory_path
    if not target_path.is_absolute():
        target_path = target_path.resolve()

    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONTACT_DIRECTORY, f, indent=2)

    return target_path


def load_contact_directory(path: Path | str | None = None) -> dict[str, dict[str, str]]:
    """Load the contact directory JSON file or return default safe contacts."""
    target_path = Path(path) if path else get_settings().contact_directory_path
    if not target_path.is_absolute():
        target_path = target_path.resolve()

    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
                if isinstance(data, list) and data:
                    normalized: dict[str, dict[str, str]] = {}
                    for item in data:
                        if isinstance(item, dict):
                            t = item.get("topic") or item.get("category") or item.get("topic_name") or "general_support"
                            normalized[str(t)] = item
                    if normalized:
                        return normalized
        except Exception:
            pass

    return dict(DEFAULT_CONTACT_DIRECTORY)


def classify_topic(question: str) -> str:
    """
    Classify question into one of the configured domain topics using keyword matching.

    Deterministic and lightweight; does not invoke any LLM.
    """
    if not question or not isinstance(question, str):
        return "general_support"

    q_lower = question.lower()

    # Match People Operations keywords
    for kw in PEOPLE_OPERATIONS_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            return "people_operations"

    # Match IT/Security keywords
    for kw in IT_SECURITY_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            return "it_security"

    # Match Employee Benefits keywords
    for kw in EMPLOYEE_BENEFITS_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            return "employee_benefits"

    # Match Travel & Expense keywords
    for kw in TRAVEL_EXPENSE_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            return "travel_expense"

    # Match Ethics & Compliance keywords
    for kw in ETHICS_COMPLIANCE_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            return "ethics_compliance"

    # Match Workplace Safety keywords
    for kw in WORKPLACE_SAFETY_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            return "workplace_safety"

    return "general_support"


def get_contact_for_topic(
    topic: str,
    directory: Any = None,
    path: Path | str | None = None,
) -> dict[str, str]:
    """
    Retrieve contact details for a given topic category.

    If no matching contact is found in the list or the directory is empty,
    creates and returns the default enterprise contact that users can email.
    """
    contacts: dict[str, dict[str, str]] = {}
    if directory is not None:
        if isinstance(directory, list):
            for item in directory:
                if isinstance(item, dict):
                    t = item.get("topic") or item.get("category") or item.get("topic_name") or "general_support"
                    contacts[str(t)] = item
        elif isinstance(directory, dict):
            contacts = dict(directory)
    else:
        contacts = load_contact_directory(path)

    # 1. Match specific topic
    contact = contacts.get(topic)
    # 2. Match general_support fallback from directory
    if not contact:
        contact = contacts.get("general_support")
    # 3. Match any available contact in directory
    if not contact and contacts:
        contact = next(iter(contacts.values()))
    # 4. If directory is empty or has no contacts in list, create default contact
    if not contact:
        contact = create_default_contact(topic)

    # Ensure all required contact fields exist and are non-empty
    res = dict(contact)
    default_ref = create_default_contact(topic)
    for field in ("name", "position", "email", "topic_name"):
        val = res.get(field)
        if not val or not str(val).strip():
            res[field] = default_ref[field]

    return res


def get_contact_for_question(
    question: str,
    directory: Any = None,
    path: Path | str | None = None,
) -> tuple[str, dict[str, str]]:
    """Determine topic for a question and return (topic, contact_info)."""
    topic = classify_topic(question)
    contact = get_contact_for_topic(topic, directory=directory, path=path)
    return topic, contact


def format_contact_footer(contact: dict[str, str]) -> str:
    """Format standard contact footer appended after normal answers and RAGAS evaluation."""
    name = contact.get("name", DEFAULT_FALLBACK_CONTACT["name"])
    position = contact.get("position", DEFAULT_FALLBACK_CONTACT["position"])
    email = contact.get("email", DEFAULT_FALLBACK_CONTACT["email"])
    return f"For more information, contact {name}, {position}, at {email}."


def format_redirect_message(topic: str, contact: dict[str, str]) -> str:
    """Format safe redirection message for queries lacking verified policy context."""
    topic_display = contact.get("topic_name") or topic.replace("_", " ")
    name = contact.get("name", DEFAULT_FALLBACK_CONTACT["name"])
    position = contact.get("position", DEFAULT_FALLBACK_CONTACT["position"])
    email = contact.get("email", DEFAULT_FALLBACK_CONTACT["email"])
    return (
        f"This question is outside the currently available policy documents. "
        f"For help with your {topic_display} query, contact {name}, {position}, at {email}."
    )
