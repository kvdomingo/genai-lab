from pydantic import BaseModel, Field

from common.chat_interfaces.langchain import LangchainChatInterface
from common.schemas import CliArguments


class Multiply(BaseModel):
    """Multiply two numbers."""

    a: int | float = Field(..., description="First number")
    b: int | float = Field(..., description="Second number")


class Add(BaseModel):
    """Add two numbers."""

    a: int | float = Field(..., description="First number")
    b: int | float = Field(..., description="Second number")


class ToolCalling(LangchainChatInterface):
    model = "mistral-nemo"
    base_prompt = None

    def __init__(self, args: CliArguments):
        self.model = args.model or self.model
        super().__init__(model=self.model, json_mode=True, tools=[Multiply, Add])
