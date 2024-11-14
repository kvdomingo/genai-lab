from collections.abc import Callable

import ollama
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from loguru import logger

from common.chat_building_blocks.io_lines import get_user_input, render_bot_pre_line
from common.settings import settings

from .base import ChatInterface


class LangchainChatInterface(ChatInterface):
    user_input_template: str
    temperature: float = 0.8
    is_tool_calling: bool = False
    base_prompt: str

    def __init__(
        self,
        model: str,
        json_mode: bool = False,
        tools: list[BaseTool] = None,
    ):
        self.model = model
        self.json_mode = json_mode
        self.tools = tools or []
        self.tool_selection = {t.name: t for t in self.tools}

        self.client = ollama.AsyncClient(str(settings.OLLAMA_URL))
        self.llm = ChatOllama(
            model=self.model,
            base_url=str(settings.OLLAMA_URL),
            temperature=self.temperature,
            format="json" if self.json_mode else "",
        )

        if len(self.tools) > 0:
            self.llm = self.llm.bind_tools(self.tools)
            self.is_tool_calling = True

        self.messages: list[BaseMessage] = []
        if self.base_prompt:
            self.messages.append(SystemMessage(self.base_prompt))

        self.format_user_input = self.user_input_formatter()

    def user_input_formatter(self) -> Callable[[str], str]:
        def func(user_input: str) -> str:
            return user_input

        return func

    async def chat(self):
        while True:
            user_input = get_user_input()
            self.messages.append(HumanMessage(self.format_user_input(user_input)))

            try:
                render_bot_pre_line()
                bot_response = ""

                if self.is_tool_calling:
                    res = await self.llm.ainvoke(self.messages)
                    for tc in res.tool_calls:
                        selected_tool = self.tool_selection[tc["name"]]
                        bot_response = await selected_tool.ainvoke(tc)
                        content = bot_response.content
                        print(content)

                        self.messages.append(bot_response)
                else:
                    async for chunk in self.llm.astream(self.messages):
                        content = chunk.content
                        bot_response += content
                        print(content, end="", flush=True)

                    self.messages.append(AIMessage(bot_response))
            except Exception as e:
                logger.error(f"An unexpected error occurred.\n\nError:\n{e}\n")
                raise
