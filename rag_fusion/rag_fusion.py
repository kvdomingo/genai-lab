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


class RagFusion:
    model: str = "llama3.2"
    embedding_model: str = "mxbai-embed-large"
    base_fusion_prompt = """
    You are a helpful assistant that generates multiple search queries based on a single
    input query.

    Generate multiple search queries related to: {question}.

    Output (4 queries):
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
        self.fusion_prompt = ChatPromptTemplate.from_template(self.base_fusion_prompt)
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
    def reciprocal_rank_fusion(documents: list[list[str]], k: int = 60):
        scores = {}

        for docs in documents:
            for rank, doc in enumerate(docs):
                doc_str = lang.dumps(doc)
                if doc_str not in scores.keys():
                    scores[doc_str] = 0

                scores[doc_str] += 1 / (rank + k)

        return [
            (lang.loads(doc), score)
            for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]

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
            self.fusion_prompt
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        retrieval_chain = (
            generate_queries | retriever.map() | self.reciprocal_rank_fusion
        )
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
