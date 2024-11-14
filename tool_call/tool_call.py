from langchain_core.tools import tool

from common.chat_interfaces.langchain import LangchainChatInterface
from common.schemas import CliArguments


@tool
def multiply(a: int | float, b: int | float) -> int | float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int | float, b: int | float) -> int | float:
    """Add two numbers."""
    return a + b


class ToolCalling(LangchainChatInterface):
    model = "llama3.2"
    base_prompt = None

    def __init__(self, args: CliArguments):
        self.model = args.model or self.model
        super().__init__(model=self.model, json_mode=True, tools=[multiply, add])
