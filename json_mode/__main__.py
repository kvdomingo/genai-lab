from common.schemas import CliArguments

from .json_mode import JSONMode

if __name__ == "__main__":
    import asyncio
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Run an AI chat bot in the terminal connecting to a locally hosted Ollama server."
    )
    parser.add_argument(
        "-m", "--model", default=None, help="Ollama model name", type=str
    )
    parsed = parser.parse_args()
    validated = CliArguments.model_validate(parsed)

    jm = JSONMode(validated)
    asyncio.run(jm())
