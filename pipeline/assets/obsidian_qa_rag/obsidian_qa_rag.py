from pathlib import Path

import dagster as dg
from chromadb import HttpClient as ChromaHttpClient
from dagster import MetadataValue
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import AsyncClient as OllamaAsyncClient

from common.pull_model import pull_model
from common.settings import settings


@dg.asset(kinds={"ollama"})
async def obsidian__embedding_model(
    ollama_client: dg.ResourceParam[OllamaAsyncClient],
) -> str:
    name = "mxbai-embed-large"
    await pull_model(ollama_client, name)
    return name


@dg.asset(kinds={"ollama", "langchain"}, deps=[obsidian__embedding_model])
def obsidian__plaintext_documents(
    context: dg.AssetExecutionContext,
) -> list[Document]:
    source_dir = Path("/opt/obsidian").resolve()
    loader = DirectoryLoader(
        str(source_dir / "Films"), glob="**/*.md", use_multithreading=True
    )
    docs = loader.load()

    context.add_output_metadata(
        {
            "count": MetadataValue.int(len(docs)),
            "sample": MetadataValue.md(docs[0].page_content),
        }
    )
    return docs


@dg.asset(kinds={"ollama", "langchain"})
def obsidian__split_documents(
    context: dg.AssetExecutionContext,
    obsidian__plaintext_documents: list[Document],
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(obsidian__plaintext_documents)

    context.add_output_metadata(
        {
            "count": MetadataValue.int(len(all_splits)),
            "sample": MetadataValue.md(all_splits[0].page_content),
        }
    )
    return all_splits


@dg.asset(kinds={"ollama", "chromadb"})
def obsidian__vector_store(
    context: dg.AssetExecutionContext,
    chroma_client: dg.ResourceParam[ChromaHttpClient],
    obsidian__split_documents: list[Document],
    obsidian__embedding_model: str,
):
    collection_name = "obsidian_qa_rag"
    Chroma.from_documents(
        documents=obsidian__split_documents,
        client=chroma_client,
        embedding=OllamaEmbeddings(
            base_url=str(settings.OLLAMA_URL),
            model=obsidian__embedding_model,
        ),
        collection_name=collection_name,
    )

    context.add_output_metadata(
        {"collection_name": MetadataValue.text(collection_name)}
    )
