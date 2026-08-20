"""Factory for creating configured Chat LLM instances."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings


def create_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """
    Instantiate and return a Chat LLM model based on current configuration.

    Supports:
      - 'gemini' via ChatGoogleGenerativeAI (requires GOOGLE_API_KEY)
      - 'openai' via ChatOpenAI (requires OPENAI_API_KEY)
      - 'ollama' via ChatOllama (uses OLLAMA_BASE_URL)
    """
    if settings is None:
        settings = get_settings()

    provider = settings.llm_provider.lower()

    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for the 'gemini' provider. "
                "Please set GOOGLE_API_KEY in your .env file."
            )
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for the 'openai' provider. "
                "Please set OPENAI_API_KEY in your .env file."
            )
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )

    if provider == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
        )

    raise ValueError(
        f"Unsupported LLM provider '{settings.llm_provider}'. "
        "Supported providers are: gemini, openai, ollama"
    )
