import sys

import bs4
import chromadb
import ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel, ConfigDict

from common.pull_model import pull_model
from common.settings import settings


class CliArguments(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chat_model: str | None
    embedding_model: str | None


class BasicQaRag:
    chat_model: str = "llama3.2"
    embedding_model: str = "mxbai-embed-large"
    vectorstore: Chroma
    retriever: VectorStoreRetriever
    base_prompt = """
    You are a helpful assistant for question-answering tasks. Use the following pieces of
    retrieved content to answer the question. If you don't know the answer, just say that
    you don't have an answer to that, don't try to make up an answer.
    Do not add any supplementary information. Do not make references to the provided text;
    give answers as if they were your own.

    Context: {context}
    """

    def __init__(self, args: CliArguments):
        self.chat_model = args.chat_model or self.chat_model
        self.embedding_model = args.embedding_model or self.embedding_model
        self.db = chromadb.HttpClient(
            host=settings.CHROMA_HOST, port=settings.CHROMA_PORT
        )
        self.client = ollama.AsyncClient(host=str(settings.OLLAMA_URL))
        self.llm = ChatOllama(base_url=str(settings.OLLAMA_URL), model=self.chat_model)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.base_prompt),
                ("human", "{input}"),
            ]
        )

    async def __call__(self):
        await self.setup()
        self.index()
        await self.generate()

    async def setup(self):
        await pull_model(self.client, self.chat_model)
        await pull_model(self.client, self.embedding_model)

    def index(self):
        bs4_strainer = bs4.SoupStrainer(
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
            web_path="https://kvd.studio/svip/ap186/image-types",
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=all_splits,
            client=self.db,
            embedding=OllamaEmbeddings(
                base_url=str(settings.OLLAMA_URL),
                model=self.embedding_model,
            ),
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6},
        )

    async def generate(self):
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

        while True:
            sources = []
            question = input("\n\nYou: ")

            if question.lower() == settings.BREAK_WORD:
                print("\n\n")
                logger.info(question)
                sys.exit(0)

            print("\n\nAI: ", end="")

            async for out in rag_chain.astream({"input": question}):
                if out.get("context"):
                    sources.append(out["context"][0])

                if out.get("answer"):
                    print(out["answer"], end="", flush=True)

            print(
                "\n\nSource: "
                + sources[0].metadata.get("source")
                + f' (l. {sources[0].metadata.get("start_index")})'
            )
