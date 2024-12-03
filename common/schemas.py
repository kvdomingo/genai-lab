from pydantic import BaseModel, ConfigDict, Field


class CliArguments(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model: str | None = Field(None)
    system_prompt: str | None = Field(None)
