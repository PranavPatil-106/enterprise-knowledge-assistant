from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.config import Settings, get_settings
from src.llm_factory import create_chat_model

SYSTEM_PROMPT_TEMPLATE = """You are a helpful enterprise assistant. Answer the user's question accurately using ONLY the context provided below.

If the context does not contain enough information to answer the question, clearly state: "I do not have enough information in the provided enterprise documents to answer this question."

Context:
{context}

Question:
{question}

Answer:"""


def format_context(documents: list[Document]) -> str:
    if not documents:
        return "No relevant context found."

    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("filename") or Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page")
        page_info = f" (page {page + 1})" if page is not None else ""
        parts.append(f"[{i}] Source: {source}{page_info}\n{doc.page_content.strip()}")

    return "\n\n".join(parts)


def get_source_names(documents: list[Document]) -> list[str]:
    sources = []
    for doc in documents:
        name = doc.metadata.get("filename") or Path(doc.metadata.get("source", "Unknown")).name
        if name and name not in sources:
            sources.append(name)
    return sources


def generate_answer(
    question: str,
    context: str | list[Document],
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> str:
    if model is None:
        model = create_chat_model(settings or get_settings())

    context_str = format_context(context) if isinstance(context, list) else str(context)
    prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_str, question=question)

    response = model.invoke([HumanMessage(content=prompt)])
    return str(response.content).strip()
