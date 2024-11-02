from common.schemas import CliArguments

from .basic_qa_rag import BasicQaRag

if __name__ == "__main__":
    import asyncio
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Run a question answering chatbot in the terminal connecting to a locally hosted Ollama server."
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Ollama model name",
        type=str,
    )
    parsed = parser.parse_args()
    validated = CliArguments.model_validate(parsed)

    qa = BasicQaRag(validated)
    asyncio.run(qa())
