"""Utility functions for credit default project."""

import os
import sys
from datetime import datetime
from typing import Any

from loguru import logger


def setup_logging(log_file: str | None = None, log_level: str = "DEBUG") -> None:
    """Set up logging configuration.

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
    """Check if the code is running in a Databricks environment.

    :return: True if running in Databricks, False otherwise.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def to_valid_date(year: int, month: int, day: int) -> datetime.date:
    """Convert given year, month, and day to a valid date.

    If the initial date is invalid, it increments the day until a valid date is found.

    :param year: The year of the date
    :param month: The month of the date
    :param day: The day of the date
    :return: A valid datetime.date object
    """
    while True:
        try:
            # Attempt to create a datetime object
            date = datetime(year, month, day)

            # If successful, convert to pandas Timestamp and return
            return date.date()

        except ValueError:
            # If the date is invalid, increment the day
            day += 1

            # If day exceeds the maximum possible for any month, reset to 1 and increment month
            if day > 31:
                day = 1
                month += 1

                # If month exceeds 12, reset to 1 and increment year
                if month > 12:
                    month = 1
                    year += 1


def normalize_arrival_date(row: dict[str, Any]) -> datetime.date:
    """Normalize the arrival date from the given row.

    This function extracts the arrival year, month, and date from the input row and converts it to a valid date.

    :param row: A dictionary containing arrival date information
    :return: A normalized datetime.date object
    """
    return to_valid_date(row["arrival_year"], row["arrival_month"], row["arrival_date"])
