from common.chat_interface import ChatInterface


class BasicBot(ChatInterface):
    def __init__(self, model: str):
        self.base_prompt = """
        You are a helpful, friendly assistant. You answer questions directly
        and concisely, and you do not include extra information unless asked.
        """.strip()

        super().__init__(model or "llama3.2")
