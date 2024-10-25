import sys
from collections.abc import Callable
from signal import SIGINT

import ollama
from colorama import Fore, Style
from loguru import logger
from tqdm import tqdm

from common.settings import settings


class ChatInterface:
    base_prompt: str
    user_input_template: str

    def __init__(self, model: str):
        self.model = model
        self.client = ollama.AsyncClient(str(settings.OLLAMA_URL))
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.base_prompt,
            }
        ]
        self.format_user_input = self.user_input_formatter()

    async def __call__(self):
        await self.setup()
        await self.chat()

    async def setup(self):
        logger.info(f"Pulling model `{self.model}`...")
        try:
            stream = await self.client.pull(self.model, stream=True)

            with tqdm(total=0) as pbar:
                async for chunk in stream:
                    if "completed" in chunk.keys() and "total" in chunk.keys():
                        pbar.total = chunk["total"]
                        pbar.update(chunk["completed"])
                    else:
                        logger.info(chunk)
        except ollama.ResponseError as e:
            logger.error(
                f"An error occurred while pulling the model `{self.model}`: {e}"
            )
            sys.exit(1)
        else:
            logger.success(f"Model {self.model} pulled")

    def user_input_formatter(self) -> Callable[[str], str]:
        def func(user_input: str) -> str:
            return user_input

        return func

    async def chat(self):
        while True:
            try:
                user_input = input("\n\n" + Fore.GREEN + "You: " + Style.RESET_ALL)
            except KeyboardInterrupt:
                sys.exit(SIGINT)

            if user_input.lower() == settings.BREAK_WORD:
                print("")
                logger.info(user_input)
                break

            self.messages.append(
                {
                    "role": "user",
                    "content": self.format_user_input(user_input),
                }
            )

            try:
                stream = await self.client.chat(
                    model=self.model, messages=self.messages, stream=True
                )
                print("\n" + Fore.BLUE + "AI: " + Style.RESET_ALL, end="")
                bot_response = ""

                async for chunk in stream:
                    content = chunk["message"]["content"]
                    bot_response += content
                    print(content, end="", flush=True)

                self.messages.append({"role": "assistant", "content": bot_response})
            except ollama.ResponseError as e:
                logger.error(
                    f"There was an error with the response. Please try again in a bit.\n\nError:\n{e}"
                )
            except Exception as e:
                logger.error(f"An unexpected error occurred.\n\nError:\n{e}")
                sys.exit(1)
