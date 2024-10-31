from pydantic import Field

from common.schemas import CliArguments as BaseCliArguments

from .json_mode import JSONMode


class CliArguments(BaseCliArguments):
    json_mode: bool = Field(False)


if __name__ == "__main__":
    import asyncio
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Run an AI chat bot in the terminal connecting to a locally hosted Ollama server."
    )
    parser.add_argument(
        "-m", "--model", default=None, help="Ollama model name", type=str
    )
    parser.add_argument(
        "-j",
        "--json-mode",
        help="Return responses in JSON format",
        default=False,
        action="store_true",
    )
    parsed = parser.parse_args()
    validated = CliArguments.model_validate(parsed)

    jm = JSONMode(validated)
    asyncio.run(jm())
