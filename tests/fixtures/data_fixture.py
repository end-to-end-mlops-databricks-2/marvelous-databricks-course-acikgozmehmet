"""Data fixture."""

import os

import pytest
from dotenv import load_dotenv

from hotel_reservations.config import Config
from hotel_reservations.data_ingestion import DataLoader
from hotel_reservations.utility import is_databricks
from tests.consts import PROJECT_DIR


@pytest.fixture(scope="function")
def dataloader() -> DataLoader:
    """Fixture to initialize and return a DataLoader instance.

    This function sets up the environment by loading configurations from a `.env` file
    and a YAML configuration file. It determines the appropriate data file path based on
    whether the code is running on Databricks or locally.

    :return: An instance of DataLoader initialized with the appropriate file path and configuration.
    """
    filename = (PROJECT_DIR / "project.env").as_posix()

    load_dotenv(filename)

    if is_databricks():
        DATA_FILE_PATH = os.environ["DATA_FILEPATH_DATABRICKS"]
    else:
        DATA_FILE_PATH = PROJECT_DIR.joinpath(os.environ["DATA_FILEPATH_LOCAL"]).as_posix()

    CONFIG_FILE_PATH = (PROJECT_DIR / "project_config.yml").as_posix()

    config = Config.from_yaml(CONFIG_FILE_PATH)
    loader = DataLoader(filepath=DATA_FILE_PATH, config=config)
    return loader
