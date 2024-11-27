import os
from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _basic_rag_corpus_factory() -> list[str]:
    return [
        "Take a leisurely walk in the park and enjoy the fresh air.",
        "Visit a local museum and discover something new.",
        "Attend a live music concert and feel the rhythm.",
        "Go for a hike and admire the natural scenery.",
        "Have a picnic with friends and share some laughs.",
        "Explore a new cuisine by dining at an ethnic restaurant.",
        "Take a yoga class and stretch your body and mind.",
        "Join a local sports league and enjoy some friendly competition.",
        "Attend a workshop or lecture on a topic you're interested in.",
        "Visit an amusement park and ride the roller coasters.",
    ]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_API_KEY: str
    OPENAI_PROJECT_ID: str

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: AnyHttpUrl = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str | None = None

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    OLLAMA_URL: AnyHttpUrl = "http://localhost:11434"
    USER_AGENT: str

    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000

    BREAK_WORD: str = "bye"
    BASE_PROMPT: str = """
    You are a friendly, helpful, general-purpose AI assistant. You answer any questions directly and concisely,
    and you do not include extra information unless asked. State if you do not know the answer; do not make
    up information.
    """
    BASIC_RAG_CORPUS: list[str] = Field(default_factory=_basic_rag_corpus_factory)


@lru_cache
def get_settings():
    settings = Settings()
    os.environ.setdefault(
        "LANGCHAIN_TRACING_V2", str(settings.LANGCHAIN_TRACING_V2).lower()
    )
    os.environ.setdefault("LANGCHAIN_ENDPOINT", str(settings.LANGCHAIN_ENDPOINT))
    os.environ.setdefault("LANGCHAIN_API_KEY", str(settings.LANGCHAIN_API_KEY))
    os.environ.setdefault("LANGCHAIN_PROJECT", str(settings.LANGCHAIN_PROJECT))
    return settings


settings = get_settings()
