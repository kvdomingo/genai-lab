from common.chat_interfaces.sdk import SDKChatInterface


class BasicBot(SDKChatInterface):
    def __init__(self, model: str, system_prompt: str = None):
        self.base_prompt = (
            system_prompt
            if system_prompt
            else """
            You are a helpful, friendly assistant. You answer questions directly
            and concisely, and you do not include extra information unless asked.
            """.strip()
        )

        super().__init__(model or "llama3.2")
