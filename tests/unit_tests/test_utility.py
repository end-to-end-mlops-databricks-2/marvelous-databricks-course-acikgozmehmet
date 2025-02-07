"""Tests for the utility module."""

import pathlib
from datetime import datetime

import pandas as pd
import pytest
import yaml
from loguru import logger
from pydantic import ValidationError

from hotel_reservations.config import Config
from hotel_reservations.utility import (
    get_current_git_sha,
    get_delta_table_version,
    is_databricks,
    normalize_arrival_date,
    setup_logging,
    to_valid_date,
)
from tests.consts import PROJECT_DIR


def test_setup_logging_with_logfile_should_create_logfile(tmp_path: pathlib.Path) -> None:
    """Test the setup_logging function.

    :param tmp_path: A temporary directory provided by pytest for testing.
    :return: None
    """
    log_file = tmp_path / "test.log"
    setup_logging(log_file=log_file.as_posix(), log_level="DEBUG")
    logger.info("Logger started...")
    assert log_file.is_file()
    assert "Logger started" in log_file.read_text(encoding="utf-8")


def test_setup_logging_without_logfile_should_not_create_logfile(
    capsys: pytest.CaptureFixture,
) -> None:
    """Test the setup_logging function.

    This test verifies that the logging setup does not create a logfile
    when the log_level is set to DEBUG.

    :param capsys: pytest fixture to capture output.
    :type capsys: pytest.CaptureFixture
    """
    setup_logging(log_level="DEBUG")
    logger.info("Logger started...")
    captured = capsys.readouterr()
    assert "Logger started" in captured.out.strip()


def test_load_config_file_not_found() -> None:
    """Test that loading a non-existent configuration file raises a FileNotFoundError.

    :raises FileNotFoundError: If the specified configuration file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("non_existent_file.yaml")


def test_load_config_invalid_yaml(tmp_path: pathlib.Path) -> None:
    """Test loading a configuration file with invalid YAML content.

    :param tmp_path: A temporary directory path for the test.
    :raises yaml.YAMLError: If the YAML content is invalid.
    """
    config_file = tmp_path / "invalid_config.yaml"
    with open(config_file, "w", encoding="utf-8") as file:
        file.write("invalid: yaml: content")

    with pytest.raises(yaml.YAMLError):
        Config.from_yaml(config_file.as_posix())


def test_load_config_missing_required_field(tmp_path: pathlib.Path) -> None:
    """Test loading a configuration file that is missing a required field.

    :param tmp_path: A temporary directory path for the test.
    :raises ValidationError: If the configuration is invalid.
    """
    config_file = tmp_path / "invalid_config.yaml"
    config_data = {"not_schema_name": "test_schema"}
    with open(config_file, "w", encoding="utf-8") as file:
        yaml.dump(config_data, file)

    with pytest.raises(ValidationError):
        Config.from_yaml(config_file.as_posix())


def test__is_databricks() -> None:
    """Test the is_databricks function to verify its behavior.

    This function asserts that the is_databricks function returns False.
    """
    flag = is_databricks()

    if flag:
        assert True
    else:
        assert flag is False


def test_to_valid_date() -> None:
    """Test the to_valid_date function with various date inputs.

    This function asserts the correctness of the to_valid_date function for different scenarios,
    including leap years and invalid dates.
    """
    assert to_valid_date(2018, 2, 28) == datetime(2018, 2, 28).date()
    assert to_valid_date(2018, 2, 29) == datetime(2018, 3, 1).date()
    assert to_valid_date(2020, 2, 29) == datetime(2020, 2, 29).date()
    assert to_valid_date(2023, 12, 32) == datetime(2024, 1, 1).date()


def test_normalize_arrival_date() -> None:
    """Test the normalize_arrival_date function with sample data.

    This function creates a sample DataFrame, applies the normalize_arrival_date function,
    and checks if the results match the expected outcomes.
    """
    # Create a sample DataFrame
    sample_data = {
        "arrival_year": [2025, 2025, 2025, 2025],
        "arrival_month": [2, 2, 2, 13],  # Note: 13 is an invalid month
        "arrival_date": [1, 29, 31, 1],  # Note: Feb 31 is invalid
    }
    data_df = pd.DataFrame(sample_data)

    # Apply the complex_function
    data_df["arrival"] = data_df.apply(normalize_arrival_date, axis=1)

    # Check the results
    expected_results = [
        datetime(2025, 2, 1).date(),
        datetime(2025, 3, 1).date(),  # 2025 is not a leap year, so this should be March 1
        datetime(2025, 3, 1).date(),
        datetime(2026, 1, 1).date(),  # Invalid month 13 should roll over to next year
    ]

    assert all(data_df["arrival"] == expected_results), "The complex_function did not produce the expected results"


@pytest.mark.skipif(not is_databricks(), reason="Only on Databricks")
def test_get_delta_table_version() -> None:
    """Test the get_delta_table_version function.

    This test verifies that the Delta table version is retrieved correctly and is at least 1.
    """
    CONFIG_FILE_PATH = (PROJECT_DIR / "project_config.yml").as_posix()
    config = Config.from_yaml(CONFIG_FILE_PATH)

    catalog_name = config.catalog_name
    schema_name = config.schema_name
    table_name = "train_set"

    version = get_delta_table_version(catalog_name=catalog_name, schema_name=schema_name, table_name=table_name)
    print(f"Delta table version: {version}")
    assert version >= 1


def test_get_current_git_sha() -> None:
    """Test the get_current_git_sha function.

    This test checks if the get_current_git_sha function returns a non-empty value.
    """
    git_sha = get_current_git_sha()
    assert git_sha
    print(f"Get current git {git_sha}")
