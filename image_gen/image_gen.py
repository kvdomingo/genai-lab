import os
from pathlib import Path
from uuid import uuid4

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from common.settings import settings


class CliArguments(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model: str | None = Field(None)
    prompt: str


class ImageGen:
    model: str = "dall-e-3"

    def __init__(self, prompt: str, model: str = None):
        self.prompt = prompt
        self.model = model or self.model
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        self.setup()

    async def __call__(self):
        res = await self.client.images.generate(prompt=self.prompt, size="1024x1024")
        url = res.data[0].url
        img = httpx.get(url).content

        with open(
            Path(__file__).resolve().parent / "outputs" / (str(uuid4()) + ".png"), "wb"
        ) as fh:
            fh.write(img)

    @staticmethod
    def setup():
        os.makedirs(Path(__file__).resolve().parent / "outputs", exist_ok=True)
