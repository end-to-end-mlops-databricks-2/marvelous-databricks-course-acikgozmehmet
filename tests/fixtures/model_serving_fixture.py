"""ModelServing fixture."""

import os

import pytest
from dotenv import load_dotenv

from hotel_reservations.config import Config
from hotel_reservations.serving import ModelServing
from hotel_reservations.utility import get_dbr_host, get_dbr_token
from tests.consts import PROJECT_DIR


@pytest.fixture(scope="function")
def model_serving() -> ModelServing:
    """Fixture to create and configure a `ModelServing` instance for testing purposes.

    This fixture loads environment variables, modifies the configuration for testing,
    and initializes a `ModelServing` with the specified configuration, tags, model, and code paths.

    :return: A configured `ModelServing` instance for testing.
    """
    filename = (PROJECT_DIR / "project.env").as_posix()
    load_dotenv(filename)

    CONFIG_FILE_PATH = (PROJECT_DIR / "project_config.yml").as_posix()
    config = Config.from_yaml(CONFIG_FILE_PATH)

    # configuration changed for testing
    config.model.name = config.model.name + "_basic"
    full_model_name = f"{config.catalog_name}.{config.schema_name}.{config.model.name}"
    endpoint_name = config.model.name.replace("_", "-") + "-test-serving"

    model = ModelServing(model_name=full_model_name, endpoint_name=endpoint_name)
    return model


@pytest.fixture(scope="module")
def deployed_model_serving(model_serving: ModelServing) -> ModelServing:
    """Deploy or update a model serving endpoint and yield the ModelServing instance.

    This fixture sets up environment variables, deploys or updates the serving endpoint,
    and cleans up by deleting the endpoint after the tests are complete.
    """
    os.environ["DBR_TOKEN"] = get_dbr_token()
    os.environ["DBR_HOST"] = get_dbr_host()
    model_serving.deploy_or_update_serving_endpoint_with_retry()

    yield model_serving

    model_serving.delete_serving_endpoint()
