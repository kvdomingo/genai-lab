from abc import ABC, abstractmethod

import ollama

from common.pull_model import pull_model


class ChatInterface(ABC):
    client: ollama.AsyncClient
    model: str

    async def __call__(self):
        await self.setup()
        await self.chat()

    async def setup(self):
        await pull_model(self.client, self.model)

    @abstractmethod
    async def chat(self):
        pass
