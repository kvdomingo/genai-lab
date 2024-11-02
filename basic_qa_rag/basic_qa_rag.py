import sys

import chromadb
import ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger

from common.chat_building_blocks.io_lines import get_user_input, render_bot_pre_line
from common.pull_model import pull_model
from common.schemas import CliArguments as BaseCliArguments
from common.settings import settings


class CliArguments(BaseCliArguments):
    collection_name: str


class BasicQaRag:
    model: str = "llama3.2"
    embedding_model: str = "mxbai-embed-large"
    base_prompt = """
    You are a helpful assistant for question-answering tasks. Use the following pieces of
    retrieved content to answer the question. If you don't know the answer, just say that
    you don't have an answer to that, don't try to make up an answer.
    Do not add any supplementary information. Do not make references to the provided text;
    give answers as if they were your own.

    Context: {context}
    """

    def __init__(self, args: CliArguments):
        self.model = args.model or self.model
        self.collection_name = args.collection_name
        self.db = chromadb.HttpClient(
            host=settings.CHROMA_HOST, port=settings.CHROMA_PORT
        )
        self.client = ollama.AsyncClient(host=str(settings.OLLAMA_URL))
        self.llm = ChatOllama(base_url=str(settings.OLLAMA_URL), model=self.model)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.base_prompt),
                ("human", "{input}"),
            ]
        )

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
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        while True:
            sources = []
            question = get_user_input()
            render_bot_pre_line()

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
