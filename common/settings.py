from functools import lru_cache

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OLLAMA_URL: AnyHttpUrl = "http://localhost:11434"
    BREAK_WORD: str = "bye"
    BASE_PROMPT: str = """
    You are a friendly, helpful, general-purpose AI assistant. You answer any questions directly and concisely,
    and you do not include extra information unless asked. State if you do not know the answer; do not make
    up information.
    """
    BASIC_RAG_CORPUS: list[str] = [
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


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()
