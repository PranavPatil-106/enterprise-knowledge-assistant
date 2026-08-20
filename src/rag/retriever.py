"""Document retrieval module for RAG knowledge base."""

from pathlib import Path
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.rag.indexer import DEFAULT_COLLECTION_NAME, get_embedding_function


def retrieve_documents(
    question: str,
    top_k: int = 4,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    settings: Settings | None = None,
) -> list[Document]:
    """
    Retrieve matching documents from the ChromaDB knowledge base for a given question.

    Uses the fixed Gemini embedding function.
    Raises FileNotFoundError if the ChromaDB directory or collection has not been indexed yet.
    """
    if settings is None:
        settings = get_settings()

    persist_dir = Path(settings.chroma_persist_directory)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB persistence directory '{persist_dir}' does not exist.\n"
            "Please build the index first by running:\n"
            "  uv run python -m src.ingest"
        )

    client = chromadb.PersistentClient(path=str(persist_dir.resolve()))
    try:
        collection = client.get_collection(name=collection_name)
        if collection.count() == 0:
            raise FileNotFoundError(
                f"ChromaDB collection '{collection_name}' exists but contains no indexed documents.\n"
                "Please build the index first by running:\n"
                "  uv run python -m src.ingest"
            )
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise FileNotFoundError(
            f"ChromaDB collection '{collection_name}' not found in '{persist_dir}'.\n"
            "Please build the index first by running:\n"
            "  uv run python -m src.ingest"
        ) from e

    embeddings = get_embedding_function(settings)
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    return vectorstore.similarity_search(query=question, k=top_k)
