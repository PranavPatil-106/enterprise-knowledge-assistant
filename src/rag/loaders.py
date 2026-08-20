"""Document loading module for RAG knowledge base."""

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.config import get_settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(directory_path: str | Path | None = None) -> list[Document]:
    """
    Load .pdf, .txt, and .md documents from the specified directory.

    Preserves document metadata such as source file paths and page numbers.
    Raises FileNotFoundError if the directory does not exist.
    Raises ValueError if no supported documents are found.
    """
    if directory_path is None:
        directory_path = get_settings().knowledge_base_path

    path = Path(directory_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {path.resolve()}")

    if not path.is_dir():
        raise NotADirectoryError(f"Knowledge base path is not a directory: {path.resolve()}")

    documents: list[Document] = []

    # Sort files for deterministic loading order
    found_files = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )

    if not found_files:
        raise ValueError(
            f"No supported documents ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) found in: {path.resolve()}"
        )

    for file_path in found_files:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        elif ext in (".txt", ".md"):
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
        else:
            continue

        for doc in docs:
            doc.metadata["filename"] = file_path.name
            doc.metadata["source"] = str(file_path.resolve())
            documents.append(doc)

    return documents
