# microbial_colony_sim/src/utils/logging_setup.py
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
    logger_name: str = "microbial_sim",
) -> logging.Logger:
    """
    Sets up a logger for the application.

    Args:
        log_level: The logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional path to a file where logs should be saved.
        log_to_console: Whether to output logs to the console.
        logger_name: The name of the logger.

    Returns:
        Configured Logger object.
    """
    logger = logging.getLogger(logger_name)

    # Prevent multiple handlers if called multiple times (e.g., in tests or reloads)
    if logger.hasHandlers():
        logger.handlers.clear()

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    )

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")  # Append mode
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Basic configuration for root logger if needed, to catch logs from libraries
    # logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    return logger


# Example of getting the logger elsewhere in the code:
# import logging
# logger = logging.getLogger("microbial_sim")
# logger.info("This is an info message.")
