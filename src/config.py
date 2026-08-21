import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"


@dataclass
class Settings:
    llm_provider: str = "gemini"
    llm_model: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    knowledge_base_path: Path = Path("data/raw")
    chroma_persist_directory: Path = Path("data/chroma")
    mcp_allowed_directory: Path = Path("data/raw")
    embedding_model: str = EMBEDDING_MODEL
    langsmith_tracing: bool = False

    @property
    def fixed_embedding_model(self) -> str:
        return self.embedding_model


def get_settings() -> Settings:
    load_dotenv(override=True)
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        knowledge_base_path=Path(os.getenv("KNOWLEDGE_BASE_PATH", "data/raw")),
        chroma_persist_directory=Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma")),
        mcp_allowed_directory=Path(os.getenv("MCP_ALLOWED_DIRECTORY", "data/raw")),
        embedding_model=EMBEDDING_MODEL,
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").strip().lower() in ("true", "1", "yes"),
    )
