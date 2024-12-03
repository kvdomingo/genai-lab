import dagster as dg
from chromadb import HttpClient as ChromaHttpClient
from dagster import MetadataValue
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_unstructured import UnstructuredLoader
from ollama import AsyncClient as OllamaAsyncClient
from unstructured.cleaners.core import clean_extra_whitespace

from common.pull_model import pull_model
from common.settings import settings


@dg.asset(kinds={"ollama"})
async def pdf__embedding_model(
    ollama_client: dg.ResourceParam[OllamaAsyncClient],
) -> str:
    name = "nomic-embed-text"
    await pull_model(ollama_client, name)
    return name


@dg.asset(kinds={"ollama", "langchain", "unstructured"}, deps=[pdf__embedding_model])
def pdf__parsed_documents(
    context: dg.AssetExecutionContext,
) -> list[Document]:
    loader = UnstructuredLoader(
        settings.BASE_DIR / "data/sample.pdf",
        post_processors=[clean_extra_whitespace],
        chunking_strategy="by_title",
    )
    docs = loader.load()

    context.add_output_metadata(
        {
            "document_count": MetadataValue.int(len(docs)),
            "document_length": MetadataValue.int(len(docs[0].page_content)),
            "sample": MetadataValue.md(docs[0].page_content),
        }
    )
    return docs


@dg.asset(kinds={"ollama", "chromadb"})
def pdf__vector_store(
    context: dg.AssetExecutionContext,
    chroma_client: dg.ResourceParam[ChromaHttpClient],
    pdf__parsed_documents: list[Document],
    pdf__embedding_model: str,
):
    collection_name = "pdf_rag"
    Chroma.from_documents(
        documents=filter_complex_metadata(pdf__parsed_documents),
        client=chroma_client,
        embedding=OllamaEmbeddings(
            base_url=str(settings.OLLAMA_URL),
            model=pdf__embedding_model,
        ),
        collection_name=collection_name,
    )

    context.add_output_metadata(
        {"collection_name": MetadataValue.text(collection_name)}
    )
