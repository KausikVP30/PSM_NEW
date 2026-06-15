from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
