import sys
from src.config import get_settings
from src.rag.indexer import build_or_rebuild_index


def main() -> None:
    settings = get_settings()
    print("Building document index in ChromaDB...")

    try:
        result = build_or_rebuild_index(settings=settings)
        print(f"Done! Indexed {result['num_chunks']} chunks from {result['num_documents']} documents.")
    except Exception as e:
        print(f"Indexing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
