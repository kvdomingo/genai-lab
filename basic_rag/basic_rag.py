from collections.abc import Callable

from common.chat_interfaces.sdk import SDKChatInterface
from common.settings import settings


class BasicRAG(SDKChatInterface):
    def __init__(self, model: str):
        self.user_input_template: str = """
        This is the recommended activity: {relevant_document}.

        The user input is: {user_input}.
        """.strip()

        self.base_prompt: str = f"""
        You are a helpful assistant that makes recommendations for activities. You answer questions directly
        and concisely, and you do not include extra information unless asked. Use a friendly tone.

        User inputs will be in the following format:
        ---
        {self.user_input_template}
        ---
        Compile a recommendation to the user based on the recommended activity and user input. Do not make
        references to the relevant document. Reply as if it were your own recommendation.
        """.strip()

        super().__init__(model or "llama3.2")

    def user_input_formatter(self) -> Callable[[str], str]:
        def func(user_input: str) -> str:
            return self.user_input_template.format(
                relevant_document=self.return_response(user_input),
                user_input=user_input,
            )

        return func

    @staticmethod
    def jaccard_similarity(query: str, document: str) -> float:
        query = set(query.lower().split(" "))
        document = set(document.lower().split(" "))
        intersection = query & document
        union = query | document
        return len(intersection) / len(union)

    def return_response(self, query: str) -> str:
        similarities = [
            self.jaccard_similarity(query, c) for c in settings.BASIC_RAG_CORPUS
        ]
        return settings.BASIC_RAG_CORPUS[similarities.index(max(similarities))]
