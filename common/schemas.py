from pydantic import BaseModel, ConfigDict, Field

from common.settings import settings


class CliArguments(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model: str = Field(settings.DEFAULT_MODEL_NAME)
