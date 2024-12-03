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
                if chunk.completed is not None and chunk.total is not None:
                    pbar.total = chunk.total
                    pbar.update(chunk.completed)
                else:
                    logger.info(chunk)
    except ollama.ResponseError as e:
        logger.error(f"An error occurred while pulling the model `{model}`: {e}")
        sys.exit(1)
    else:
        logger.success(f"Model `{model}` pulled")
