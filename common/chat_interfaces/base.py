import sys
from abc import ABC, abstractmethod

import ollama
from loguru import logger
from tqdm import tqdm


class ChatInterface(ABC):
    client: ollama.AsyncClient
    model: str

    async def __call__(self):
        await self.setup()
        await self.chat()

    async def setup(self):
        logger.info(f"Pulling model `{self.model}`...")
        try:
            stream = await self.client.pull(self.model, stream=True)

            with tqdm(total=0) as pbar:
                async for chunk in stream:
                    if "completed" in chunk.keys() and "total" in chunk.keys():
                        pbar.total = chunk["total"]
                        pbar.update(chunk["completed"])
                    else:
                        logger.info(chunk)
        except ollama.ResponseError as e:
            logger.error(
                f"An error occurred while pulling the model `{self.model}`: {e}"
            )
            sys.exit(1)
        else:
            logger.success(f"Model {self.model} pulled")

    @abstractmethod
    async def chat(self):
        pass
