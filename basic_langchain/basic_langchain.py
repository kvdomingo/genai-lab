from common.chat_interfaces.langchain import LangchainChatInterface
from common.schemas import CliArguments


class BasicLangchain(LangchainChatInterface):
    model: str = "llama3.2"
    base_prompt: str = "You are a helpful, friendly assistant."

    def __init__(self, args: CliArguments):
        self.model = args.model or self.model
        super().__init__(self.model)
