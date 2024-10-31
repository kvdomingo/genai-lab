from common.schemas import CliArguments

from .tool_call import ToolCalling

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

    tc = ToolCalling(validated)
    asyncio.run(tc())
