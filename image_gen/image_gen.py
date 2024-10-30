import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
from openai import AsyncOpenAI
from pydantic import UUID4, BaseModel, ConfigDict, Field

from common.settings import settings


class CliArguments(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model: str | None = Field(None)
    prompt: str


class Output(BaseModel):
    id: UUID4
    prompt: str


class ImageGen:
    model: str = "dall-e-3"
    outputs_dir: Path = Path(__file__).resolve().parent / "outputs"
    outputs: list[Output] = []

    def __init__(self, prompt: str, model: str = None):
        self.prompt = prompt
        self.model = model or self.model
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            project=settings.OPENAI_PROJECT_ID,
        )

        self.setup()

    async def __call__(self):
        res = await self.client.images.generate(prompt=self.prompt, size="1024x1024")
        url = res.data[0].url

        async with httpx.AsyncClient() as client:
            img = (await client.get(url)).content

        id = uuid4()

        with open(self.outputs_dir / f"{id}.png", "wb") as fh:
            fh.write(img)

        self.outputs.append(Output(id=id, prompt=self.prompt))

        with open(self.outputs_dir / "outputs.json", "w+") as fh:
            json.dump([o.model_dump(mode="json") for o in self.outputs], fh, indent=2)

    def setup(self):
        os.makedirs(self.outputs_dir, exist_ok=True)

        if not (self.outputs_dir / "outputs.json").exists():
            with open(self.outputs_dir / "outputs.json", "w+") as fh:
                json.dump([], fh, indent=2)
        else:
            with open(self.outputs_dir / "outputs.json") as fh:
                self.outputs = [Output.model_validate(j) for j in json.load(fh)]
