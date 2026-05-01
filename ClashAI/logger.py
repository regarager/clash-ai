from colorama import Fore, Style


def info(msg: str):
    print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {msg}")


def debug(msg: str):
    print(f"{Fore.GREEN}[DEBUG]{Style.RESET_ALL} {msg}")


def warn(msg: str):
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {msg}")


def error(msg: str):
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")


def header(msg: str) -> str:
    return f"===== {msg} ====="
