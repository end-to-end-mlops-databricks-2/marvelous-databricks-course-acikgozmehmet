"""Utility functions for credit default project."""

import os
import sys

from loguru import logger


def setup_logging(log_file: str | None = None, log_level: str = "DEBUG") -> None:
    """
    Set up logging configuration.

    This function configures the logger to write logs to a specified file and/or to standard output.

    :param log_file: Optional log file path to write logs to. If None, logs will not be written to a file.
    :param log_level: The logging level to use. Default is 'DEBUG'.
    """
    # Remove the default handler
    logger.remove()

    # Add file logger with rotation if log_file is given
    if log_file:
        logger.add(log_file, rotation="500 MB", level=log_level)

    # Add stdout logger
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )


def is_databricks() -> bool:
    """
    Check if the code is running in a Databricks environment.

    :return: True if running in Databricks, False otherwise.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ
