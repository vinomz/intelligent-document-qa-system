import logging
import os
from logging.handlers import RotatingFileHandler
from config import settings

def get_logger(name: str = "app", log_dir: str = "/app/logs", log_level: str = settings.DEFAULT_LOGLEVEL) -> logging.Logger:
    """
    Create and return a configured logger.
    Supports:
      - Console logging
      - Rotating file logging
    """

    # Ensure log folder exists
    os.makedirs(log_dir, exist_ok=True)

    # Log format
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create logger
    logger = logging.getLogger(name)
    level = getattr(logging, log_level.upper(), settings.DEFAULT_LOGLEVEL)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger is reused
    if logger.handlers:
        return logger

    # ---------------------------
    # 1. Console Handler
    # ---------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)

    # ---------------------------
    # 2. File Handler (Rotating)
    # ---------------------------
    file_path = os.path.join(log_dir, f"{name}.log")

    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB per log file
        backupCount=5              # keep last 5 files
    )
    file_handler.setLevel(level)

    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
