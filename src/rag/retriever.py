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
    cfg = settings or get_settings()
    persist_dir = Path(cfg.chroma_persist_directory)

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory '{persist_dir}' does not exist. Run 'python -m src.ingest' first."
        )

    client = chromadb.PersistentClient(path=str(persist_dir.resolve()))
    try:
        collection = client.get_collection(name=collection_name)
        if collection.count() == 0:
            raise FileNotFoundError(f"ChromaDB collection '{collection_name}' is empty. Run 'python -m src.ingest'.")
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise FileNotFoundError(f"Collection '{collection_name}' not found in '{persist_dir}'.") from e

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embedding_function(cfg),
    )

    return vectorstore.similarity_search(query=question, k=top_k)
