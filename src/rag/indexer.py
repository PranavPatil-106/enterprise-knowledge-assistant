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
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def get_embedding_function(settings: Settings | None = None) -> GoogleGenerativeAIEmbeddings:
    cfg = settings or get_settings()
    if not cfg.google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing in .env file.")

    return GoogleGenerativeAIEmbeddings(
        model=cfg.embedding_model,
        google_api_key=cfg.google_api_key,
    )


def build_or_rebuild_index(
    documents_path: str | Path | None = None,
    persist_directory: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    settings: Settings | None = None,
) -> dict[str, Any]:
    cfg = settings or get_settings()
    doc_path = Path(documents_path or cfg.knowledge_base_path)
    persist_dir = Path(persist_directory or cfg.chroma_persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    raw_documents = load_documents(doc_path)
    chunks = create_chunks(raw_documents)
    embeddings = get_embedding_function(cfg)

    client = chromadb.PersistentClient(path=str(persist_dir.resolve()))
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        client=client,
    )

    return {
        "collection_name": collection_name,
        "num_documents": len(raw_documents),
        "num_chunks": len(chunks),
        "persist_directory": str(persist_dir.resolve()),
        "vectorstore": vectorstore,
    }
