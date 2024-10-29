from .image_gen import CliArguments, ImageGen

if __name__ == "__main__":
    import asyncio
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate an image using DALL-E.")
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="OpenAI image generation model name",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="The image prompt/instructions",
        type=str,
    )
    parsed = parser.parse_args()

    ig = ImageGen(**CliArguments.model_validate(parsed).model_dump())
    asyncio.run(ig())
