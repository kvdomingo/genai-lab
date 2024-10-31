from collections.abc import Callable
from inspect import signature

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from common.chat_interfaces.sdk import SDKChatInterface
from common.settings import settings


class SemanticRAG(SDKChatInterface):
    transformer: SentenceTransformer
    doc_embeddings: signature(SentenceTransformer.encode).return_annotation
    threshold = 0.3

    def __init__(self, model: str):
        self.transformer_model = model

        self.user_input_template: str = """
        These are potential activities:
        {relevant_document}

        The user query is: {user_input}
        """.strip()

        self.base_prompt: str = f"""
        You are a helpful assistant that makes recommendations for activities. You answer questions directly
        and concisely, and you do not include extra information unless asked. Use a friendly tone.

        User inputs will be in the following format:
        ---
        {self.user_input_template}
        ---
        Provide the user with 2 recommended activities based on their query. Do not make
        references to the relevant document. Reply as if it were your own recommendation.
        """.strip()

        super().__init__("llama3.2")

    async def setup(self):
        await super().setup()
        self.transformer = SentenceTransformer(
            self.transformer_model or "all-MiniLM-L6-v2"
        )
        self.doc_embeddings = self.transformer.encode(settings.BASIC_RAG_CORPUS)

    def user_input_formatter(self) -> Callable[[str], str]:
        def func(user_input: str) -> str:
            sorted_index = self.get_similarity_scores(user_input)
            recommended_docs = [
                settings.BASIC_RAG_CORPUS[val]
                for val, score in sorted_index
                if score > self.threshold
            ]
            return self.user_input_template.format(
                relevant_document="\n".join(recommended_docs),
                user_input=user_input,
            )

        return func

    def get_similarity_scores(self, query: str) -> list[tuple[int, float]]:
        query_embedding = self.transformer.encode([query])
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)
        indexed = list(enumerate(similarities[0]))
        sorted_index = sorted(indexed, key=lambda x: x[1], reverse=True)
        return sorted_index
