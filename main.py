import ollama
from colorama import Fore, Style
from loguru import logger
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434"

MODEL_NAME = "llama3.2:1b"

BREAK_WORD = "bye"

BASE_PROMPT = """
You are a friendly, helpful, general-purpose AI assistant. You answer any questions directly and concisely,
and you do not include extra information unless asked.
"""


async def main():
    messages = [
        {
            "role": "system",
            "content": BASE_PROMPT,
        }
    ]

    client = ollama.AsyncClient(OLLAMA_URL)
    logger.info(f"Pulling model {MODEL_NAME}...")
    stream = await client.pull(MODEL_NAME, stream=True)

    with tqdm(total=0) as pbar:
        async for chunk in stream:
            if "completed" in chunk.keys() and "total" in chunk.keys():
                pbar.total = chunk["total"]
                pbar.update(chunk["completed"])
            else:
                logger.info(chunk)

    logger.success(f"Model {MODEL_NAME} pulled")

    while True:
        user_input = input("\n\n" + Fore.GREEN + "You: " + Style.RESET_ALL)

        if user_input.lower() == BREAK_WORD:
            logger.info("\n" + user_input)
            break

        messages.append({"role": "user", "content": user_input})

        try:
            stream = await client.chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            )
            print("\n" + Fore.BLUE + "AI: " + Style.RESET_ALL, end="")
            bot_response = ""

            async for chunk in stream:
                content = chunk["message"]["content"]
                bot_response += content
                print(content, end="", flush=True)

            messages.append({"role": "assistant", "content": bot_response})
        except ollama.ResponseError as e:
            logger.error(
                f"There was an error with the response. Please try again in a bit.\n\nError:\n{e}"
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred.\n\nError:\n{e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
