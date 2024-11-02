import sys
from collections.abc import Callable

import ollama
from loguru import logger

from common.chat_building_blocks.io_lines import (
    get_user_input,
    render_bot_pre_line,
)
from common.settings import settings

from .base import ChatInterface


class SDKChatInterface(ChatInterface):
    user_input_template: str
    base_prompt: str

    def __init__(self, model: str):
        self.model = model
        self.client = ollama.AsyncClient(str(settings.OLLAMA_URL))
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self.base_prompt}
        ]
        self.format_user_input = self.user_input_formatter()

    def user_input_formatter(self) -> Callable[[str], str]:
        def func(user_input: str) -> str:
            return user_input

        return func

    async def chat(self):
        while True:
            user_input = get_user_input()
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
                render_bot_pre_line()
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
