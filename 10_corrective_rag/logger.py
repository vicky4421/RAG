import inspect
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def get_logger(
    log_name: str | None = None,
    log_file: str | None = None,
    cli_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Returns a logger configured for the calling file.
    If log_file is not provided, it creates a .log file named after the caller
    (e.g., basic_rag.py -> logs/basic_rag.log).
    """
    # 1. Determine caller file path dynamically
    current_dir = Path(__file__).resolve().parent
    if log_name is None:
        caller_frame = inspect.stack()[1]  # func or script executed get_logger()
        caller_file = Path(
            caller_frame.filename
        )  # extract path for script executed get_logger()
        log_name = caller_file.stem  # file name without extension

    # Determine target log file
    if log_file is None:
        log_dir = current_dir / "logs"
        log_dir.mkdir(
            parents=True, exist_ok=True
        )  # checks if log_dir exists or creates new
        resolved_log_path = (
            log_dir / f"{caller_file.stem}.log"
        )  # caller_file scope is get_logger() and not just if block
    else:
        resolved_log_path = current_dir / log_file
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(level=logging.DEBUG)

    # prevent duplicate handlers, if get_logger() called already
    if logger.handlers:
        return logger

    # 3 Rich cli output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    rich_handler.setLevel(level=cli_level)
    logger.addHandler(hdlr=rich_handler)

    # 4. File Handler (Isolated per-script log file)
    file_handler = logging.FileHandler(
        filename=resolved_log_path, mode="a", encoding="utf-8"
    )
    file_handler.setLevel(level=file_level)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(hdlr=file_handler)

    logger.propagate = False
    return logger
