"""BasicModel fixture."""

import mlflow
import pytest
from dotenv import load_dotenv

from hotel_reservations.basic_model import BasicModel
from hotel_reservations.config import Config, Tags
from tests.consts import PROJECT_DIR


@pytest.fixture(scope="function")
def basic_model() -> BasicModel:
    """Fixture to provide a BasicModel instance for testing.

    This fixture loads environment variables from a `.env` file, reads configuration
    from a YAML file, and initializes a BasicModel object with the configuration
    and a testing tag.

    :return: An instance of BasicModel configured for testing.
    """
    filename = (PROJECT_DIR / "project.env").as_posix()

    load_dotenv(filename)

    CONFIG_FILE_PATH = (PROJECT_DIR / "project_config.yml").as_posix()

    config = Config.from_yaml(CONFIG_FILE_PATH)

    # configuration changed for testing
    config.experiment_name = config.experiment_name + "-testing"
    config.model.name = config.model.name + "_testing"
    config.model.artifact_path = config.model.artifact_path + "-testing"

    tags = Tags(branch="testing")
    basic_model = BasicModel(config=config, tags=tags)
    return basic_model


@pytest.fixture(scope="function")
def logged_basic_model(basic_model: BasicModel) -> BasicModel:  # Generator[BasicModel, None, None]:
    """Set up and log a basic model for testing.

    This fixture prepares a BasicModel instance, logs it to MLflow, and cleans up after the test.

    :param basic_model: An instance of BasicModel to be prepared and logged.
    :return: A generator yielding the prepared BasicModel instance.
    """
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")  # noqa
    basic_model.load_data()
    basic_model.prepare_features()
    basic_model.train()

    experiment = mlflow.get_experiment_by_name(basic_model.experiment_name)
    assert experiment is None

    basic_model.log_model()

    yield basic_model

    experiment = mlflow.get_experiment_by_name(basic_model.experiment_name)
    assert experiment is not None

    print(f"{experiment.artifact_location} = ")

    mlflow.delete_experiment(experiment.experiment_id)  # noqa
    assert experiment is None
