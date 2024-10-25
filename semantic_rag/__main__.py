from loguru import logger
from sentence_transformers import SentenceTransformer

from common.schemas import CliArguments
from common.settings import settings


class SemanticRAG:
    def __init__(self, args: CliArguments):
        self.model = args.model or "all-MiniLM-L6-v2"
        self.transformer = SentenceTransformer(self.model)
        self.embeddings = self.transformer.encode(settings.BASIC_RAG_CORPUS)

    async def __call__(self, *args, **kwargs):
        logger.info(self.embeddings)


if __name__ == "__main__":
    import asyncio
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Run an AI chat bot in the terminal connecting to a locally hosted Ollama server."
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="HuggingFace sentence transformer model name",
        type=str,
    )
    parsed = parser.parse_args()

    rag = SemanticRAG(CliArguments.model_validate(parsed))
    asyncio.run(rag())
