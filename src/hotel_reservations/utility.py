"""Utility functions for credit default project."""

import os
import pathlib
import sys
from datetime import datetime
from typing import Any

from delta.tables import DeltaTable
from loguru import logger
from pyspark.sql import SparkSession


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


def get_delta_table_version(
    catalog_name: str, schema_name: str, table_name: str, spark: SparkSession | None = None
) -> int:
    """Retrieve the current version of a Delta table.

    This function fetches the version of a specified Delta table using the provided catalog, schema, and table names.

    :param catalog_name: The name of the catalog containing the table
    :param schema_name: The name of the schema containing the table
    :param table_name: The name of the Delta table
    :param spark: An optional SparkSession instance
    :return: The current version of the Delta table, or None if an error occurs
    """
    # Create or get existing SparkSession
    if spark is None:
        spark = SparkSession.builder.getOrCreate()

    # Construct the full table name
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

    try:
        # Get the Delta table
        delta_table = DeltaTable.forName(spark, full_table_name)

        # Get the current version
        current_version = delta_table.history().select("version").first()[0]

        return current_version

    except Exception as e:
        print(f"Error getting version for table {full_table_name}: {str(e)}")
        return None


def get_current_git_sha() -> str:
    """Retrieve the current Git commit SHA from a file.

    This function reads the Git commit SHA from a file named 'commit_sha.txt' located in the 'data' directory.

    :return: The Git commit SHA as a string
    :raises FileNotFoundError: If the commit_sha.txt file does not exist
    """
    file_path = pathlib.Path(__file__).parent / "data" / "commit_sha.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path.as_posix()} does not exist")

    git_sha = file_path.read_text(encoding="utf-8")
    return git_sha.strip()
