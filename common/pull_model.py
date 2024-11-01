import sys

import ollama
from loguru import logger
from tqdm import tqdm


async def pull_model(client: ollama.AsyncClient, model: str):
    logger.info(f"Pulling model `{model}`...")
    try:
        stream = await client.pull(model, stream=True)

        with tqdm(total=0) as pbar:
            async for chunk in stream:
                if "completed" in chunk.keys() and "total" in chunk.keys():
                    pbar.total = chunk["total"]
                    pbar.update(chunk["completed"])
                else:
                    logger.info(chunk)
    except ollama.ResponseError as e:
        logger.error(f"An error occurred while pulling the model `{model}`: {e}")
        sys.exit(1)
    else:
        logger.success(f"Model `{model}` pulled")
