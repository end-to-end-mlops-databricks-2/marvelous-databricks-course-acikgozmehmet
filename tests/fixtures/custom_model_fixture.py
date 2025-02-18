"""BasicModel fixture."""

import json

import mlflow
import mlflow.entities
import pandas as pd
import pytest
from dotenv import load_dotenv

from hotel_reservations.config import Config, Tags
from hotel_reservations.custom_model import CustomModel, load_model
from hotel_reservations.tracking import validate_experiment_deleted
from tests.consts import PROJECT_DIR


@pytest.fixture(scope="function")
def custom_model() -> CustomModel:
    """Fixture to create and configure a `CustomModel` instance for testing purposes.

    This fixture loads environment variables, modifies the configuration for testing,
    and initializes a `CustomModel` with the specified configuration, tags, model, and code paths.

    :return: A configured `CustomModel` instance for testing.
    """
    filename = (PROJECT_DIR / "project.env").as_posix()
    load_dotenv(filename)

    CONFIG_FILE_PATH = (PROJECT_DIR / "project_config.yml").as_posix()
    config = Config.from_yaml(CONFIG_FILE_PATH)
    # configuration changed for testing
    config.experiment_name = config.experiment_name + "-custom-testing"
    config.model.name = config.model.name + "_custom_testing"
    config.model.artifact_path = config.model.artifact_path + "-custom-testing"

    tags = Tags(branch="custom-model-testing")

    model_path = (PROJECT_DIR / "tests" / "test_data" / "lightgbm-pipeline-model" / "model.pkl").resolve()
    model = load_model(model_path.as_posix())

    code_paths = []
    custom_model = CustomModel(config=config, tags=tags, model=model, code_paths=code_paths)

    return custom_model


@pytest.fixture(scope="function")
def logged_custom_model(custom_model: CustomModel) -> CustomModel:
    """Log a custom model to MLflow and yield it for testing.

    This fixture sets up MLflow tracking, logs the model, and cleans up after the test.

    :param custom_model: The custom model to be logged
    :return: The logged custom model
    """
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")  # noqa

    experiment = mlflow.get_experiment_by_name(custom_model.experiment_name)
    assert experiment is None

    input_data_json_string = """{"columns": ["no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
    "required_car_parking_space", "lead_time", "repeated_guest", "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
    "avg_price_per_room", "no_of_special_requests", "type_of_meal_plan", "room_type_reserved"],
    "data": [[2, 0, 2, 2, 0, 12, 0, 0, 0, 76.29, 0, "Not Selected", "Room_Type 1"]]}"""
    input_data = json.loads(input_data_json_string)
    model_input = pd.DataFrame(data=input_data["data"], columns=input_data["columns"])
    model_output = pd.DataFrame([{"prediction": 1}])

    custom_model.log_model(model_input=model_input, model_output=model_output)

    yield custom_model

    experiment = mlflow.get_experiment_by_name(custom_model.experiment_name)
    assert experiment is not None

    mlflow.delete_experiment(experiment.experiment_id)  # noqa
    validate_experiment_deleted(experiment)  # noqa
