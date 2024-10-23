import sys

import ollama
from colorama import Fore, Style
from loguru import logger
from tqdm import tqdm

from llm_lab.schemas import CliArguments
from llm_lab.settings import settings


async def main(args: CliArguments):
    messages = [
        {
            "role": "system",
            "content": settings.BASE_PROMPT,
        }
    ]

    client = ollama.AsyncClient(str(settings.OLLAMA_URL))
    logger.info(f"Pulling model {args.model}...")

    try:
        stream = await client.pull(args.model, stream=True)

        with tqdm(total=0) as pbar:
            async for chunk in stream:
                if "completed" in chunk.keys() and "total" in chunk.keys():
                    pbar.total = chunk["total"]
                    pbar.update(chunk["completed"])
                else:
                    logger.info(chunk)
    except ollama.ResponseError as e:
        logger.error(f"An error occurred while pulling the model: {e}")
        sys.exit(1)
    else:
        logger.success(f"Model {args.model} pulled")

    while True:
        user_input = input("\n\n" + Fore.GREEN + "You: " + Style.RESET_ALL)

        if user_input.lower() == settings.BREAK_WORD:
            logger.info(user_input)
            break

        messages.append({"role": "user", "content": user_input})

        try:
            stream = await client.chat(
                model=args.model,
                messages=messages,
                stream=True,
            )
            print("\n" + Fore.BLUE + "AI: " + Style.RESET_ALL, end="")
            bot_response = ""

            async for chunk in stream:
                content = chunk["message"]["content"]
                bot_response += content
                print(content, end="", flush=True)

            messages.append({"role": "assistant", "content": bot_response})
        except ollama.ResponseError as e:
            logger.error(
                f"There was an error with the response. Please try again in a bit.\n\nError:\n{e}"
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred.\n\nError:\n{e}")
            sys.exit(1)


if __name__ == "__main__":
    import asyncio
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Run an AI chat bot in the terminal connecting to a locally hosted Ollama server."
    )
    parser.add_argument(
        "-m",
        "--model",
        default=settings.DEFAULT_MODEL_NAME,
        help="Ollama model name",
        type=str,
    )
    args = parser.parse_args()

    asyncio.run(main(CliArguments.model_validate(args)))
