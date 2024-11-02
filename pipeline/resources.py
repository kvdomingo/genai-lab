from enum import Enum

import chromadb
import ollama

from common.settings import settings


class Resource(Enum):
    OLLAMA = "ollama_client"
    CHROMA = "chroma_client"


RESOURCES = {
    Resource.OLLAMA.value: ollama.AsyncClient(host=str(settings.OLLAMA_URL)),
    Resource.CHROMA.value: chromadb.HttpClient(
        host=settings.CHROMA_HOST, port=settings.CHROMA_PORT
    ),
}
