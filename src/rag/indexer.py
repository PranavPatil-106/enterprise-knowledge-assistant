"""Indexing and ChromaDB storage module for RAG knowledge base."""

from pathlib import Path
from typing import Any
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings, get_settings
from src.rag.loaders import load_documents

DEFAULT_COLLECTION_NAME = "enterprise_knowledge"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def create_chunks(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def get_embedding_function(settings: Settings | None = None) -> GoogleGenerativeAIEmbeddings:
    """
    Create Gemini embeddings using the fixed embedding model and configured API key.
    """
    if settings is None:
        settings = get_settings()

    if not settings.google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing or empty. Please set GOOGLE_API_KEY in your .env file."
        )

    # Fixed non-configurable Gemini embedding model from Settings
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def build_or_rebuild_index(
    documents_path: str | Path | None = None,
    persist_directory: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Loads documents, splits into chunks, and saves to ChromaDB using fixed Gemini embeddings.
    Replaces only the specified collection so rebuilding is safe and avoids duplicates.
    """
    if settings is None:
        settings = get_settings()

    doc_path = Path(documents_path) if documents_path is not None else settings.knowledge_base_path
    persist_dir = Path(persist_directory) if persist_directory is not None else settings.chroma_persist_directory

    # Ensure persistence directory exists
    persist_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load documents
    raw_documents = load_documents(doc_path)
    num_documents = len(raw_documents)

    # 2. Chunk documents
    chunks = create_chunks(raw_documents)
    num_chunks = len(chunks)

    # 3. Initialize embeddings
    embeddings = get_embedding_function(settings)

    # 4. Safe rebuild: delete existing collection if it exists
    client = chromadb.PersistentClient(path=str(persist_dir.resolve()))
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        # Collection didn't exist yet, normal for first run
        pass

    # 5. Index chunks in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        client=client,
    )

    return {
        "collection_name": collection_name,
        "num_documents": num_documents,
        "num_chunks": num_chunks,
        "persist_directory": str(persist_dir.resolve()),
        "vectorstore": vectorstore,
    }
