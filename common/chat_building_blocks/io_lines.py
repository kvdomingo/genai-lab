import sys
from signal import SIGINT

from colorama import Fore, Style
from loguru import logger

from common.settings import settings


def get_user_input() -> str:
    try:
        user_input = input("\n\n" + Fore.GREEN + "You: " + Style.RESET_ALL)
    except KeyboardInterrupt:
        sys.exit(SIGINT)

    if user_input.lower().strip() == settings.BREAK_WORD:
        print("")
        logger.info(user_input)
        sys.exit(0)

    return user_input


def render_bot_pre_line():
    print("\n" + Fore.BLUE + "AI: " + Style.RESET_ALL, end="")
