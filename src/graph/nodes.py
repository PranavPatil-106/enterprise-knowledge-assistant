"""Workflow node functions for the LangGraph agent graph."""

import re
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

from src.answering import format_context, generate_answer, get_source_names
from src.config import get_settings
from src.contact_directory import (
    format_redirect_message,
    get_contact_for_question,
    get_contact_for_topic,
)
from src.evaluation import evaluate_answer
from src.github_mcp_client import fetch_github_context_for_question, is_github_question
from src.graph.state import GraphState
from src.llm_factory import create_chat_model
from src.mcp_client import read_text_file_via_mcp
from src.rag.retriever import retrieve_documents_with_scores


def retriever_node(state: GraphState) -> dict[str, Any]:
    """Retriever Agent node: fetches matching documents from ChromaDB, GitHub MCP, or reads top source via Filesystem MCP."""
    question = state["question"]
    settings = get_settings()

    # Route repository / code inquiries through official GitHub Remote MCP
    if is_github_question(question):
        print("[Node: Retriever Agent] Fetching live repository information through GitHub MCP...")
        mcp_context, source_label = fetch_github_context_for_question(question, settings=settings)
        doc = Document(page_content=mcp_context, metadata={"source": source_label})
        return {
            "documents": [doc],
            "context": mcp_context,
            "mcp_context": mcp_context,
            "sources": [source_label],
            "has_verified_context": True,
            "contact": get_contact_for_topic("general_support"),
            "contact_topic": "general_support",
            "is_redirect": False,
            "used_github_mcp": True,
        }

    print("[Node: Retriever Agent] Fetching relevant documents from ChromaDB...")
    topic, contact = get_contact_for_question(question)

    doc_scores = retrieve_documents_with_scores(question=question, settings=settings)

    # Check relevance against minimum threshold
    has_verified = False
    documents = []
    if doc_scores:
        top_doc, top_score = doc_scores[0]
        if top_score >= settings.min_retrieval_score:
            has_verified = True
            # Retain relevant documents meeting or near threshold
            documents = [doc for doc, score in doc_scores if score >= settings.min_retrieval_score]

    if not has_verified:
        return {
            "documents": [],
            "context": "",
            "mcp_context": "",
            "sources": [],
            "has_verified_context": False,
            "contact": contact,
            "contact_topic": topic,
            "is_redirect": True,
            "used_github_mcp": False,
        }

    context = format_context(documents)
    sources = get_source_names(documents)
    mcp_context = ""

    print("[Node: Retriever Agent] Reading top source through Filesystem MCP...")
    top_doc = documents[0]
    top_source = top_doc.metadata.get("source") or top_doc.metadata.get("file_name") or ""

    if top_source:
        source_path = Path(top_source)
        if not source_path.is_absolute():
            candidate = settings.mcp_allowed_directory / source_path.name
            if candidate.exists():
                target_file = candidate
            else:
                target_file = settings.mcp_allowed_directory / source_path
        else:
            target_file = source_path

        try:
            mcp_text = read_text_file_via_mcp(
                file_path=target_file,
                allowed_directory=settings.mcp_allowed_directory,
            )
            mcp_context = mcp_text
            context = (
                f"{context}\n\n"
                f"=== Additional Content via Filesystem MCP Server ({target_file.name}) ===\n"
                f"{mcp_text}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Filesystem MCP retrieval failed for top document '{target_file.name}': {e}"
            ) from e

    return {
        "documents": documents,
        "context": context,
        "mcp_context": mcp_context,
        "sources": sources,
        "has_verified_context": True,
        "contact": contact,
        "contact_topic": topic,
        "is_redirect": False,
        "used_github_mcp": False,
    }


INSUFFICIENT_CONTEXT_PATTERNS = [
    r"not have enough information",
    r"does not contain (?:enough |any )?information",
    r"do not contain (?:enough |any )?information",
    r"cannot be answered from the provided",
    r"can(?:not|'t) answer this question (?:based on|from) the provided",
    r"not mentioned in the provided",
    r"no information provided",
    r"outside the (?:scope|currently available policy documents)",
    r"is not covered in the provided",
    r"neither the provided documents nor",
    r"not addressed in the provided",
]


def is_insufficient_context_answer(answer: str) -> bool:
    """Check if the generated answer indicates that the query cannot be answered from policy context."""
    if not answer or not isinstance(answer, str) or not answer.strip():
        return True
    a_lower = answer.lower()
    for pattern in INSUFFICIENT_CONTEXT_PATTERNS:
        if re.search(pattern, a_lower):
            return True
    return False


def response_node(state: GraphState) -> dict[str, Any]:
    """Response Agent node: generates a grounded answer or a fixed contact redirect."""
    question = state["question"]
    has_verified = state.get("has_verified_context", False)

    if not has_verified:
        print("[Node: Response Agent] No verified context found. Preparing contact redirect...")
        topic = state.get("contact_topic", "general_support")
        contact = state.get("contact")
        if not contact:
            contact = get_contact_for_topic(topic)
        redirect_msg = format_redirect_message(topic, contact)
        return {
            "answer": redirect_msg,
            "is_redirect": True,
            "contact": contact,
            "sources": [],
        }

    print("[Node: Response Agent] Generating grounded answer from context...")
    context = state.get("context", "")
    settings = get_settings()
    model = create_chat_model(settings)
    answer = generate_answer(question=question, context=context, model=model, settings=settings)

    if is_insufficient_context_answer(answer):
        print("[Node: Response Agent] Insufficient context in documents. Preparing contact redirect...")
        topic = state.get("contact_topic", "general_support")
        contact = state.get("contact")
        if not contact:
            contact = get_contact_for_topic(topic)
        redirect_msg = format_redirect_message(topic, contact)
        return {
            "answer": redirect_msg,
            "has_verified_context": False,
            "is_redirect": True,
            "contact": contact,
            "sources": [],
        }

    return {
        "answer": answer,
        "is_redirect": False,
    }


def evaluator_node(state: GraphState) -> dict[str, Any]:
    """Evaluator Agent node: evaluates response quality with RAGAS when verified context exists."""
    has_verified = state.get("has_verified_context", False)

    if not has_verified or state.get("is_redirect", False):
        print("[Node: Evaluator Agent] Skipping RAGAS because no verified policy context was found.")
        return {
            "evaluation": "Not evaluated because no verified policy context was found.",
        }

    print("[Node: Evaluator Agent] Evaluating response quality with RAGAS...")

    question = state["question"]
    answer = state.get("answer", "")
    documents = state.get("documents", [])

    evaluation_result = evaluate_answer(
        question=question,
        answer=answer,
        documents=documents,
    )

    return {
        "evaluation": evaluation_result,
    }
