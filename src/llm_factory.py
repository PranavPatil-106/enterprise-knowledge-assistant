from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings


def create_chat_model(settings: Settings | None = None) -> BaseChatModel:
    cfg = settings or get_settings()
    provider = cfg.llm_provider.lower()

    if provider == "gemini":
        if not cfg.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required in .env for Gemini.")
        return ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            google_api_key=cfg.google_api_key,
            temperature=0.0,
            max_retries=3,
        )

    if provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required in .env for OpenAI.")
        return ChatOpenAI(
            model=cfg.llm_model,
            api_key=cfg.openai_api_key,
            temperature=0.0,
            max_retries=3,
        )

    if provider == "ollama":
        return ChatOllama(
            model=cfg.llm_model,
            temperature=0.0,
        )

    raise ValueError(f"Unsupported LLM provider '{cfg.llm_provider}'. Use: gemini, openai, or ollama.")
