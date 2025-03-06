"""Utility functions for credit default project."""

import argparse
import json
import os
import pathlib
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import requests
from delta.tables import DeltaTable
from loguru import logger
from pyspark.dbutils import DBUtils
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
    If the file doesn't exist, it returns a formatted datetime string instead.

    :param: None
    :return: The Git commit SHA as a string or a formatted datetime string
    :raises FileNotFoundError: If the commit_sha.txt file does not exist and cannot be read
    """
    file_path = pathlib.Path(__file__).parent / "data" / "commit_sha.txt"
    formatted_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    git_sha = formatted_datetime if not file_path.exists() else file_path.read_text(encoding="utf-8")
    return git_sha.strip()


def call_endpoint(endpoint_name: str, records: list[dict]) -> tuple[int, str]:
    """Call a serving endpoint with the given endpoint name and record data.

    :param endpoint_name: The name of the serving endpoint to call
    :param records: A list of dictionaries containing the data to send to the endpoint
    :return: A tuple containing the response status code and text
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": records},
    )
    return response.status_code, response.text


def get_dbr_token() -> str:
    """Retrieve the Databricks API token.

    This function obtains the API token from the Databricks notebook context.

    :return: The Databricks API token as a string.
    :raises ValueError: If not running in a Databricks environment.
    Important note: Never use your personal databricks token in real application. Create Service Principal instead.
    This is just for testing purposes
    """
    if is_databricks():
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        return dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    else:
        raise ValueError("This function is only supported on Databricks.")


def get_dbr_host() -> str:
    """Retrieve the Databricks workspace URL.

    This function obtains the workspace URL from Spark configuration.

    :return: The Databricks workspace URL as a string.
    :raises ValueError: If not running in a Databricks environment.
    """
    if is_databricks():
        spark = SparkSession.builder.getOrCreate()
        return spark.conf.get("spark.databricks.workspaceUrl")
    else:
        raise ValueError("This function is only supported on Databricks.")


def create_parser(args: Sequence[str] = None) -> argparse.Namespace:
    """Create and configure an argument parser for MLOps on Databricks.

    This function sets up a parser with subparsers for different MLOps operations.

    :param args: Optional sequence of command-line argument strings
    :return: Parsed argument namespace
    """
    parser = argparse.ArgumentParser(description="Parser for MLOps on Databricks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--root_path", type=str, required=True, help="Path of root on DAB")
    common_args.add_argument("--env", type=str, required=True, help="Path of env file on DAB")

    # Data ingestion subparser
    subparsers.add_parser("data_ingestion", parents=[common_args], help="Data ingestion options")

    # Model training & registering subparser
    model_parser = subparsers.add_parser(
        "model_train_register", parents=[common_args], help="Model training and registering options"
    )
    model_parser.add_argument("--git_sha", type=str, required=True, help="git sha of the commit")
    model_parser.add_argument("--job_run_id", type=str, required=True, help="job run_id of the mlflow")
    model_parser.add_argument("--branch", type=str, required=True, help="branch of the project")

    # Deployment subparser
    subparsers.add_parser("deployment", parents=[common_args], help="Deployment options")

    return parser.parse_args(args)


def dict_to_json_to_dict(input_data: dict | list[dict]) -> tuple[str, dict | list[dict]]:
    """Convert a dictionary or list of dictionaries to JSON and then back to the original format.

    :param input_data (dict or list of dict): The input data to be converted.
    :return: tuple: A tuple containing (json_string, output_data)
                    json_string: A JSON-formatted string representation of the input data
                    output_data: Data converted back from the JSON string (dict or list of dict).
    :raises TypeError: If there's an error during conversion
    """
    try:
        # Convert dict to JSON string
        json_string = json.dumps(input_data, default=str, indent=2)
        # Convert JSON string back to dict
        output_data = json.loads(json_string)

        return json_string, output_data
    except (TypeError, json.JSONDecodeError) as e:
        raise TypeError(f"Error: {str(e)}") from e
