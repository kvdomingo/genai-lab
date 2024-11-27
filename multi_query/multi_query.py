import sys
from operator import itemgetter

import chromadb
import langchain.load as lang
import ollama
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger

from common.chat_building_blocks.io_lines import get_user_input, render_bot_pre_line
from common.pull_model import pull_model
from common.schemas import CliArguments as BaseCliArguments
from common.settings import settings


class CliArguments(BaseCliArguments):
    collection_name: str


class MultiQuery:
    model: str = "llama3.2"
    embedding_model: str = "mxbai-embed-large"
    base_retrieval_prompt = """
    You are an AI language model assistant. Your task is to generate 5 different
    versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is
    to help the user overcome some of the limitations of distance-based similarity
    search. Provide these alternative questions separated by newlines.

    Original question: {question}
    """
    base_qa_prompt: str = """
    Answer the following question based on this context:

    {context}

    Question: {question}
    """

    def __init__(self, args: CliArguments):
        self.model = args.model or self.model
        self.collection_name = args.collection_name
        self.db = chromadb.HttpClient(
            host=settings.CHROMA_HOST, port=settings.CHROMA_PORT
        )
        self.client = ollama.AsyncClient(host=str(settings.OLLAMA_URL))
        self.llm = ChatOllama(
            base_url=str(settings.OLLAMA_URL),
            model=self.model,
            temperature=0,
        )
        self.retrieval_prompt = ChatPromptTemplate.from_template(
            self.base_retrieval_prompt
        )
        self.qa_prompt = ChatPromptTemplate.from_template(self.base_qa_prompt)

    async def __call__(self):
        await self.setup()
        await self.chat()

    async def setup(self):
        coll = self.db.get_collection(self.collection_name)
        peek = coll.peek(1)
        if len(peek.get("documents", [])) == 0:
            logger.error(
                f"Collection `{self.collection_name}` is empty or does not exist."
            )
            sys.exit(1)

        await pull_model(self.client, self.model)
        await pull_model(self.client, self.embedding_model)

    @staticmethod
    def get_unique_union(documents: list[list[str]]):
        flat = [lang.dumps(d) for doc in documents for d in doc]
        unique = list(set(flat))
        return [lang.loads(doc) for doc in unique]

    async def chat(self):
        vector_store = Chroma(
            client=self.db,
            collection_name=self.collection_name,
            embedding_function=OllamaEmbeddings(
                base_url=str(settings.OLLAMA_URL),
                model=self.embedding_model,
            ),
        )
        retriever = vector_store.as_retriever()
        generate_queries = (
            self.retrieval_prompt
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        retrieval_chain = generate_queries | retriever.map() | self.get_unique_union
        rag_chain = (
            {
                "context": retrieval_chain,
                "question": itemgetter("question"),
            }
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

        while True:
            question = get_user_input()
            render_bot_pre_line()

            async for out in rag_chain.astream({"question": question}):
                print(out, end="", flush=True)
