from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.config import get_settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(directory_path: str | Path | None = None) -> list[Document]:
    path = Path(directory_path or get_settings().knowledge_base_path)

    if not path.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {path.resolve()}")
    if not path.is_dir():
        raise NotADirectoryError(f"Knowledge base path is not a directory: {path.resolve()}")

    files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
    if not files:
        raise ValueError(f"No supported documents found in: {path.resolve()}")

    documents = []
    for file_path in files:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            docs = PyPDFLoader(str(file_path)).load()
        elif ext in (".txt", ".md"):
            docs = TextLoader(str(file_path), encoding="utf-8").load()
        else:
            continue

        for doc in docs:
            doc.metadata["filename"] = file_path.name
            doc.metadata["source"] = str(file_path.resolve())
            documents.append(doc)

    return documents
