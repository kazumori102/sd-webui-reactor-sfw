"""ReActor logging controls.

Do not replace InsightFace classes: its 1.0.1 detector/router and prepare API
must remain available to ReActor and every other extension in this process.
"""
import logging
from scripts.reactor_logger import logger


def apply_logging_patch(console_logging_level):
    levels = {0: logging.WARNING, 1: logging.INFO + 5, 2: logging.INFO}
    logger.setLevel(levels.get(console_logging_level, logging.INFO))
