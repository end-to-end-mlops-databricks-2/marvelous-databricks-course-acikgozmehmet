import os

import pytest
from dotenv import load_dotenv

from credit_default.config import Config
from credit_default.data_ingestion import DataLoader
from credit_default.utility import is_databricks
from tests.consts import PROJECT_DIR


@pytest.fixture(scope="function")
def dataloader() -> DataLoader:
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
