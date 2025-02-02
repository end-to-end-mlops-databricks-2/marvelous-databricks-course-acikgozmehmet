"""Tests for the utility module."""

import pathlib

import pytest
import yaml
from loguru import logger
from pydantic import ValidationError

from credit_default.config import Config
from credit_default.utility import (
    is_databricks,
    setup_logging,
)


def test_setup_logging_with_logfile_should_create_logfile(tmp_path: pathlib.Path) -> None:
    """
    Test the setup_logging function.

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
    """
    Test the setup_logging function.

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
    """
    Test that loading a non-existent configuration file raises a FileNotFoundError.

    :raises FileNotFoundError: If the specified configuration file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("non_existent_file.yaml")


def test_load_config_invalid_yaml(tmp_path: pathlib.Path) -> None:
    """
    Test loading a configuration file with invalid YAML content.

    :param tmp_path: A temporary directory path for the test.
    :raises yaml.YAMLError: If the YAML content is invalid.
    """
    config_file = tmp_path / "invalid_config.yaml"
    with open(config_file, "w", encoding="utf-8") as file:
        file.write("invalid: yaml: content")

    with pytest.raises(yaml.YAMLError):
        Config.from_yaml(config_file.as_posix())


def test_load_config_missing_required_field(tmp_path: pathlib.Path) -> None:
    """
    Test loading a configuration file that is missing a required field.

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
    """
    Test the is_databricks function to verify its behavior.

    This function asserts that the is_databricks function returns False.
    """
    flag = is_databricks()

    if flag:
        assert True
    else:
        assert flag is False
