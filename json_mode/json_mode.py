from common.chat_interfaces.langchain import LangchainChatInterface
from common.schemas import CliArguments


class JSONMode(LangchainChatInterface):
    model = "llama3.2"
    base_prompt: str = """
    You are a helpful, friendly assistant. The user will input a sentence or phrase.
    You must analyze the text and extract entities such as names, dates, and locations.
    Respond using JSON only, with the following keys: name, date, location.
    These keys must always be present. If entities cannot be extracted from the user input,
    then the value should be null. Dates should be formatted as YYYY-MM-DD.
    """

    def __init__(self, args: CliArguments):
        self.model = args.model or self.model
        super().__init__(self.model)
