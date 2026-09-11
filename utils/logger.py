import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(
    log_name: str = "app.log",
    cli_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Standard production-style logger.
    Directs all output to the centralized ROOT/logs directory.
    """

    logger = logging.getLogger(log_name)
    logger.setLevel(level=logging.DEBUG)

    # prevent duplicate handlers, if get_logger() called already
    if logger.handlers:
        return logger

    # Rich cli output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
        log_time_format="%d/%m/%Y %I:%M:%S %p",
    )
    rich_handler.setLevel(level=cli_level)
    logger.addHandler(hdlr=rich_handler)

    # Centralized file handler
    log_path = LOG_DIR / log_name
    file_handler = logging.FileHandler(filename=log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level=file_level)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S %p",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(hdlr=file_handler)

    logger.propagate = False
    return logger
