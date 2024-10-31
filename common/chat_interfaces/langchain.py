import sys
from collections.abc import Callable
from signal import SIGINT
from typing import Literal

import ollama
from colorama import Fore, Style
from langchain_ollama import ChatOllama
from loguru import logger
from pydantic import BaseModel

from common.settings import settings

from .base import ChatInterface


class LangchainChatInterface(ChatInterface):
    user_input_template: str
    temperature: float = 0.8
    disable_streaming: bool = False

    def __init__(
        self,
        model: str,
        json_mode: bool = False,
        tools: list[BaseModel | Callable] = None,
    ):
        self.model = model
        self.json_mode = json_mode
        self.tools = tools or []

        self.client = ollama.AsyncClient(str(settings.OLLAMA_URL))
        self.llm = ChatOllama(
            model=self.model,
            base_url=str(settings.OLLAMA_URL),
            temperature=self.temperature,
            format="json" if self.json_mode else "",
        )

        if len(self.tools) > 0:
            self.llm.bind_tools(self.tools)
            self.disable_streaming = True

        self.messages: list[tuple[Literal["system", "human", "assistant"], str]] = [
            ("system", self.base_prompt),
        ]
        self.format_user_input = self.user_input_formatter()

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

            self.messages.append(("human", self.format_user_input(user_input)))

            try:
                print("\n" + Fore.BLUE + "AI: " + Style.RESET_ALL, end="")
                bot_response = ""

                if self.disable_streaming:
                    res = await self.llm.ainvoke(self.messages)
                    bot_response = res.content
                    print(bot_response)
                else:
                    async for chunk in self.llm.astream(self.messages):
                        content = chunk.content
                        bot_response += content
                        print(content, end="", flush=True)

                self.messages.append(("assistant", bot_response))
            except Exception as e:
                logger.error(f"An unexpected error occurred.\n\nError:\n{e}")
                sys.exit(1)
