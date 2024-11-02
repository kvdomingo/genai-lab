import chromadb
import dagster as dg
import ollama
from bs4 import SoupStrainer
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.pull_model import pull_model
from common.settings import settings


@dg.asset
async def embedding_model(ollama_client: dg.ResourceParam[ollama.AsyncClient]) -> str:
    name = "mxbai-embed-large"
    await pull_model(ollama_client, name)
    return name


@dg.asset(deps=[embedding_model])
def plaintext_documents(context: dg.AssetExecutionContext) -> list[Document]:
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

    context.add_output_metadata({"count": len(docs)})
    return docs


@dg.asset
def split_documents(
    context: dg.AssetExecutionContext,
    plaintext_documents: list[Document],
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(plaintext_documents)

    context.add_output_metadata({"count": len(all_splits)})
    return all_splits


@dg.asset
def vector_store(
    chroma_client: dg.ResourceParam[chromadb.HttpClient],
    split_documents: list[Document],
    embedding_model: str,
):
    Chroma.from_documents(
        documents=split_documents,
        client=chroma_client,
        embedding=OllamaEmbeddings(
            base_url=str(settings.OLLAMA_URL),
            model=embedding_model,
        ),
        collection_name="basic_qa_rag",
    )
