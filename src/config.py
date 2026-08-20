"""Configuration management for Enterprise Knowledge Assistant."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Fixed embedding model (not configurable via environment)
FIXED_EMBEDDING_MODEL: str = "gemini-embedding-001"
VALID_LLM_PROVIDERS: set[str] = {"gemini", "openai", "ollama"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    google_api_key: str
    openai_api_key: str
    ollama_base_url: str
    knowledge_base_path: Path
    chroma_persist_directory: Path
    mcp_allowed_directory: Path
    langsmith_tracing: bool
    langsmith_api_key: str
    langsmith_project: str
    langsmith_endpoint: str

    def __post_init__(self) -> None:
        if not self.llm_provider or self.llm_provider.lower() not in VALID_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider: '{self.llm_provider}'. "
                f"Supported providers are: {', '.join(sorted(VALID_LLM_PROVIDERS))}"
            )
        if not self.llm_model or not self.llm_model.strip():
            raise ValueError("LLM_MODEL is required and cannot be empty.")

    @property
    def embedding_model(self) -> str:
        """Fixed Gemini embedding model (read-only)."""
        return FIXED_EMBEDDING_MODEL

    @property
    def fixed_embedding_model(self) -> str:
        """Alias for fixed Gemini embedding model."""
        return FIXED_EMBEDDING_MODEL

    @classmethod
    def from_env(cls) -> "Settings":
        """Load and validate settings from environment variables."""
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip()

        return cls(
            llm_provider=provider,
            llm_model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
            knowledge_base_path=Path(os.getenv("KNOWLEDGE_BASE_PATH", "data/raw")),
            chroma_persist_directory=Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma")),
            mcp_allowed_directory=Path(os.getenv("MCP_ALLOWED_DIRECTORY", "data/raw")),
            langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").strip().lower() in ("true", "1", "yes"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", "").strip(),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "enterprise-knowledge-assistant").strip(),
            langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").strip(),
        )


def get_settings() -> Settings:
    """Helper function to load settings from environment."""
    return Settings.from_env()
