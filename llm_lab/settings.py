from functools import lru_cache

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OLLAMA_URL: AnyHttpUrl = "http://localhost:11434"
    DEFAULT_MODEL_NAME: str = "llama3.2:1b"
    BREAK_WORD: str = "bye"
    BASE_PROMPT: str = """
    You are a friendly, helpful, general-purpose AI assistant. You answer any questions directly and concisely,
    and you do not include extra information unless asked.
    """


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()
