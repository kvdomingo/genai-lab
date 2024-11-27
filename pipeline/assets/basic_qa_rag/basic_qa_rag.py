import dagster as dg
from bs4 import SoupStrainer
from chromadb import HttpClient as ChromaHttpClient
from dagster import MetadataValue
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import AsyncClient as OllamaAsyncClient

from common.pull_model import pull_model
from common.settings import settings


@dg.asset
async def portfolio__embedding_model(
    ollama_client: dg.ResourceParam[OllamaAsyncClient],
) -> str:
    name = "mxbai-embed-large"
    await pull_model(ollama_client, name)
    return name


@dg.asset(deps=[portfolio__embedding_model])
def portfolio__plaintext_documents(
    context: dg.AssetExecutionContext,
) -> list[Document]:
    web_paths = (
        "https://kvd.studio/svip/ap186/basic-video",
        "https://kvd.studio/svip/ap186/blob-analysis",
        "https://kvd.studio/svip/ap186/practical-2",
        "https://kvd.studio/svip/ap186/image-segmentation",
        "https://kvd.studio/svip/ap186/image-types",
        "https://kvd.studio/svip/ap186/measuring-area",
    )

    bs4_strainer = SoupStrainer(
        name=(
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "a",
            "ul",
            "ol",
            "li",
            "div",
            "code",
            "caption",
            "figcaption",
            "table",
            "thead",
            "tbody",
            "tr",
            "td",
            "th",
        ),
    )
    loader = WebBaseLoader(
        web_paths=web_paths,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    context.add_output_metadata(
        {
            "count": MetadataValue.int(len(docs)),
            "sample": MetadataValue.text(docs[0].page_content),
        }
    )
    return docs


@dg.asset
def portfolio__split_documents(
    context: dg.AssetExecutionContext,
    portfolio__plaintext_documents: list[Document],
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(portfolio__plaintext_documents)

    context.add_output_metadata(
        {
            "count": MetadataValue.int(len(all_splits)),
            "sample": MetadataValue.text(all_splits[0].page_content),
        }
    )
    return all_splits


@dg.asset
def portfolio__vector_store(
    context: dg.AssetExecutionContext,
    chroma_client: dg.ResourceParam[ChromaHttpClient],
    portfolio__split_documents: list[Document],
    portfolio__embedding_model: str,
):
    collection_name = "basic_qa_rag"
    Chroma.from_documents(
        documents=portfolio__split_documents,
        client=chroma_client,
        embedding=OllamaEmbeddings(
            base_url=str(settings.OLLAMA_URL),
            model=portfolio__embedding_model,
        ),
        collection_name=collection_name,
    )

    context.add_output_metadata(
        {"collection_name": MetadataValue.text(collection_name)}
    )
