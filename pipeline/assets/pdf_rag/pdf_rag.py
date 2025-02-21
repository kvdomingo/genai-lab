from pathlib import Path

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
from pipeline.assets.pdf_rag.partitions import subdirectory_partitions


@dg.asset(kinds={"ollama"})
async def pdf__embedding_model(
    ollama_client: dg.ResourceParam[OllamaAsyncClient],
) -> str:
    name = "nomic-embed-text"
    await pull_model(ollama_client, name)
    return name


@dg.asset(
    kinds={"ollama", "langchain", "unstructured"},
    deps=[pdf__embedding_model],
    partitions_def=subdirectory_partitions,
)
def pdf__parsed_documents(context: dg.AssetExecutionContext) -> list[Document]:
    subdirectory = context.partition_key
    raw_docs: list[Path] = list((settings.RAG_ROOT_DIR / subdirectory).glob("*.pdf"))
    loaded_docs = []

    for i, doc in enumerate(raw_docs):
        context.log.info(f"Loading document {i+1}/{len(raw_docs)}: {doc.name}")

        loader = UnstructuredLoader(
            doc,
            post_processors=[clean_extra_whitespace],
            chunking_strategy="by_title",
        )
        loaded_docs.extend(loader.load())

    context.add_output_metadata(
        {
            "file_count": MetadataValue.int(len(raw_docs)),
            "document_count": MetadataValue.int(len(loaded_docs)),
            "document_length": 0
            if len(loaded_docs) == 0
            else MetadataValue.int(len(loaded_docs[0].page_content)),
            "sample": None
            if len(loaded_docs) == 0
            else MetadataValue.md(loaded_docs[0].page_content),
        }
    )
    return loaded_docs


@dg.asset(kinds={"ollama", "chromadb"}, partitions_def=subdirectory_partitions)
def pdf__vector_store(
    context: dg.AssetExecutionContext,
    chroma_client: dg.ResourceParam[ChromaHttpClient],
    pdf__parsed_documents: list[Document],
    pdf__embedding_model: str,
):
    if len(pdf__parsed_documents) == 0:
        return

    collection_name = context.partition_key
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
