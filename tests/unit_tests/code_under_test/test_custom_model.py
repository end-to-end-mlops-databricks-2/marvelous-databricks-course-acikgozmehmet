"""Unit tests for the custom model."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from loguru import logger

from hotel_reservations.config import Config
from hotel_reservations.custom_model import CustomModel, ModelWrapper, load_model
from hotel_reservations.tracking import search_registered_model_versions
from hotel_reservations.utility import is_databricks
from tests.consts import PROJECT_DIR


def test_load_model() -> None:
    """Test the loading of a LightGBM pipeline model.

    Ensures that the model is successfully loaded from the specified file path.
    """
    filepath = (PROJECT_DIR / "tests" / "test_data" / "lightgbm-pipeline-model" / "model.pkl").resolve()
    model = load_model(filepath.as_posix())
    assert model is not None


def test_model_wrapper_init() -> None:
    """Test the initialization of the ModelWrapper class.

    Ensures that the ModelWrapper class is initialized correctly.
    """
    model_path = (PROJECT_DIR / "tests" / "test_data" / "lightgbm-pipeline-model" / "model.pkl").resolve()
    model = load_model(model_path.as_posix())
    model_wrapper = ModelWrapper(model=model)
    assert isinstance(model_wrapper, ModelWrapper)
    assert model_wrapper.model is not None


def test_model_wrapper_predict() -> None:
    """Test the initialization of the ModelWrapper class.

    Ensures that the ModelWrapper class is initialized correctly.
    """
    model_path = (PROJECT_DIR / "tests" / "test_data" / "lightgbm-pipeline-model" / "model.pkl").resolve()
    model = load_model(model_path.as_posix())
    model_wrapper = ModelWrapper(model=model)
    input_data = pd.read_csv(
        (PROJECT_DIR / "tests" / "test_data" / "train_test_pred" / "xtest.csv").resolve().as_posix()
    ).head(1)
    predictions = model_wrapper.predict(context=None, model_input=input_data)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == 1


def test_custom_model_init(custom_model: CustomModel) -> None:
    """Test the initialization of the CustomModel class.

    This function verifies that a CustomModel instance is correctly initialized
    with the provided configuration, tags, model, and code paths. It also ensures
    that the attributes of the CustomModel instance are of the expected types.
    """
    assert isinstance(custom_model.config, Config)
    assert isinstance(custom_model, CustomModel)
    assert isinstance(custom_model.model, ModelWrapper)


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_custom_model_log_model_success(logged_custom_model: CustomModel) -> None:
    """Tests the logging of a custom model using the `CustomModel` class.

    This function sets up a custom model configuration, loads a test model,
    and validates that the experiment is created with the expected artifact location.
    It also ensures proper cleanup by deleting the experiment after validation.

    :raises AssertionError: If the experiment is not created or its artifact location is missing.
    """
    # validate the experiment exists and has the expected artifact location
    experiment = mlflow.get_experiment_by_name(logged_custom_model.experiment_name)
    assert experiment is not None
    assert experiment.artifact_location


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_custom_model_register_success(logged_custom_model: CustomModel) -> None:
    """Tests the registration of a custom model using the `CustomModel` class.

    This function sets up a custom model configuration, loads a test model,
    registers the model, and validates that it is registered.
    """
    # experiment = mlflow.get_experiment_by_name(logged_basic_model.experiment_name) # noqa
    logged_custom_model.register_model()

    model_name = f"{logged_custom_model.catalog_name}.{logged_custom_model.schema_name}.{logged_custom_model.model_name}"  # Replace with your model's full name
    registered_models = search_registered_model_versions(full_model_name=model_name)
    assert registered_models

    if registered_models:
        logger.info(f"Model '{model_name}' is registered.")
        for mv in registered_models:
            logger.info(f"Name: {mv.name}")
            logger.info(f"Version: {mv.version}")
            logger.info(f"Stage: {mv.current_stage}")
            logger.info(f"Description: {mv.description}")

        # delete the mess
        # delete_registered_model(model_name=model_name)  # noqa
    else:
        logger.info(f"Model '{model_name}' is not registered.")


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_load_latest_model_and_predict_on_databricks(logged_custom_model: CustomModel) -> None:
    """Test load_latest_model_and_predict."""
    logged_custom_model.register_model()
    reg_custom_model = logged_custom_model

    input = pd.read_csv(
        (PROJECT_DIR / "tests" / "test_data" / "train_test_pred" / "xtest.csv").resolve().as_posix()
    ).head(10)

    predictions = reg_custom_model.load_latest_model_and_predict(input_data=input)

    logger.info(f"{predictions = }")
    assert len(predictions) > 0
    assert len(predictions) == 10

    #  clean up the mess
    model_name = (
        f"{logged_custom_model.catalog_name}.{logged_custom_model.schema_name}.{logged_custom_model.model_name}"
    )
    registered_models = search_registered_model_versions(full_model_name=model_name)  # noqa

    # if registered_models:
    #     delete_registered_model(model_name=model_name)  #  noqa
