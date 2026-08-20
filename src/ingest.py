"""CLI entry point to build or rebuild the RAG knowledge index."""

import sys
from src.config import get_settings
from src.rag.indexer import build_or_rebuild_index


def main() -> None:
    print("=" * 60)
    print("Enterprise Knowledge Assistant - Document Ingestion & Indexing")
    print("=" * 60)

    try:
        settings = get_settings()
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)

    print(f"Knowledge Base Path:     {settings.knowledge_base_path}")
    print(f"ChromaDB Persist Dir:    {settings.chroma_persist_directory}")
    print(f"Fixed Embedding Model:   {settings.embedding_model}")
    print("-" * 60)

    try:
        print("Loading documents and indexing into ChromaDB...")
        result = build_or_rebuild_index(settings=settings)
        print("\n[SUCCESS] Indexing complete!")
        print(f"  - Collection name:   {result['collection_name']}")
        print(f"  - Documents loaded:  {result['num_documents']}")
        print(f"  - Chunks indexed:    {result['num_chunks']}")
        print(f"  - Storage directory: {result['persist_directory']}")
    except ValueError as e:
        print(f"\n[ERROR] Configuration or validation error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[ERROR] File error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Indexing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
