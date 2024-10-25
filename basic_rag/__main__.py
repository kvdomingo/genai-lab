import asyncio
import sys

import ollama
from colorama import Fore, Style
from loguru import logger
from tqdm import tqdm

from common.schemas import CliArguments
from common.settings import settings


class BasicRAG:
    def __init__(self, args: CliArguments):
        self.args = args
        self.user_input_template: str = """
        This is the recommended activity: {relevant_document}.

        The user input is: {user_input}.
        """
        self.base_prompt: str = f"""
        You are a helpful assistant that makes recommendations for activities. You answer questions directly
        and concisely, and you do not include extra information unless asked. Use a friendly tone.

        User inputs will be in the following format:
        ---
        {self.user_input_template}
        ---
        Compile a recommendation to the user based on the recommended activity and user input. Do not make
        references to the relevant document. Reply as if it were your own recommendation.
        """
        self.corpus: list[str] = [
            "Take a leisurely walk in the park and enjoy the fresh air.",
            "Visit a local museum and discover something new.",
            "Attend a live music concert and feel the rhythm.",
            "Go for a hike and admire the natural scenery.",
            "Have a picnic with friends and share some laughs.",
            "Explore a new cuisine by dining at an ethnic restaurant.",
            "Take a yoga class and stretch your body and mind.",
            "Join a local sports league and enjoy some friendly competition.",
            "Attend a workshop or lecture on a topic you're interested in.",
            "Visit an amusement park and ride the roller coasters.",
        ]
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.base_prompt,
            }
        ]

    async def __call__(self):
        client = ollama.AsyncClient(str(settings.OLLAMA_URL))
        logger.info(f"Pulling model {self.args.model}...")

        try:
            stream = await client.pull(self.args.model, stream=True)

            with tqdm(total=0) as pbar:
                async for chunk in stream:
                    if "completed" in chunk.keys() and "total" in chunk.keys():
                        pbar.total = chunk["total"]
                        pbar.update(chunk["completed"])
                    else:
                        logger.info(chunk)
        except ollama.ResponseError as e:
            logger.error(f"An error occurred while pulling the model: {e}")
            sys.exit(1)
        else:
            logger.success(f"Model {self.args.model} pulled")

        while True:
            user_input = input("\n\n" + Fore.GREEN + "You: " + Style.RESET_ALL)

            if user_input.lower() == settings.BREAK_WORD:
                logger.info(user_input)
                break

            self.messages.append(
                {
                    "role": "user",
                    "content": self.user_input_template.format(
                        relevant_document=self.return_response(user_input),
                        user_input=user_input,
                    ),
                }
            )

            try:
                stream = await client.chat(
                    model=self.args.model,
                    messages=self.messages,
                    stream=True,
                )
                print("\n" + Fore.BLUE + "AI: " + Style.RESET_ALL, end="")
                bot_response = ""

                async for chunk in stream:
                    content = chunk["message"]["content"]
                    bot_response += content
                    print(content, end="", flush=True)

                self.messages.append({"role": "assistant", "content": bot_response})
            except ollama.ResponseError as e:
                logger.error(
                    f"There was an error with the response. Please try again in a bit.\n\nError:\n{e}"
                )
            except Exception as e:
                logger.error(f"An unexpected error occurred.\n\nError:\n{e}")
                sys.exit(1)

    @staticmethod
    def jaccard_similarity(query: str, document: str) -> float:
        query = set(query.lower().split(" "))
        document = set(document.lower().split())
        intersection = query & document
        union = query | document
        return len(intersection) / len(union)

    def return_response(self, query: str) -> str:
        similarities = [self.jaccard_similarity(query, c) for c in self.corpus]
        return self.corpus[similarities.index(max(similarities))]


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Run an AI chat bot in the terminal connecting to a locally hosted Ollama server."
    )
    parser.add_argument(
        "-m",
        "--model",
        default=settings.DEFAULT_MODEL_NAME,
        help="Ollama model name",
        type=str,
    )
    parsed = parser.parse_args()

    rag = BasicRAG(CliArguments.model_validate(parsed))

    asyncio.run(rag())
