"""Unit tests for basic_models."""

import mlflow
import pytest
from pyspark.sql import SparkSession

from hotel_reservations.basic_model import BasicModel
from hotel_reservations.utility import is_databricks
from src.hotel_reservations.tracking import delete_registered_model, search_registered_model_versions


def test_basic_model_fixture_as_expected(basic_model: BasicModel) -> None:
    """Test the basic model fixture to ensure it meets expected criteria.

    This function checks various attributes and configurations of the basic model fixture.

    :param basic_model: The basic model fixture to be tested
    """
    assert isinstance(basic_model, BasicModel)
    assert basic_model.config.experiment_name == "/Shared/hotel-reservations-basic-testing"
    assert basic_model.config.model.name == "hotel_reservations_model_basic_testing"
    assert basic_model.config.model.artifact_path == "lightgbm-pipeline-model-basic-testing"
    assert basic_model.tags["git_sha"] is not None
    assert basic_model.tags["branch"] == "testing"


@pytest.mark.skipif(is_databricks(), reason="Only Local")
def test_load_data_fail(basic_model: BasicModel) -> None:
    """Test the failure case of the load_data method in BasicModel.

    This test verifies that a ValueError is raised when attempting to load data outside of Databricks.
    """
    with pytest.raises(ValueError) as exc:
        basic_model.load_data()
    assert "This function is only supported on Databricks." in str(exc.value)


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_load_data_on_databricks(basic_model: BasicModel) -> None:
    """Test the data loading functionality of the BasicModel.

    This function verifies that the BasicModel can successfully load data and
    that the loaded datasets have the expected properties.
    """
    basic_model.load_data()

    assert basic_model.train_set.shape[0] > 1
    assert basic_model.test_set.shape[0] > 1
    assert basic_model.data_version

    assert basic_model.X_train.shape[0] > 1
    assert basic_model.X_test.shape[0] > 1
    assert basic_model.y_train.shape[0] > 1
    assert basic_model.y_test.shape[0] > 1


def test_prepare_features(basic_model: BasicModel) -> None:
    """Test the prepare_features method in BasicModel."""
    basic_model.prepare_features()

    assert basic_model.preprocessor is not None
    assert basic_model.pipeline is not None


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_train_on_databricks(basic_model: BasicModel) -> None:
    """Perform end-to-end testing of the BasicModel on Databricks.

    This test function loads data, prepares features, and trains the model,
    asserting various model attributes before and after training.

    :param basic_model: An instance of the BasicModel class to be tested
    """
    basic_model.load_data()
    basic_model.prepare_features()

    assert not hasattr(basic_model.pipeline, "n_features_in_")
    assert not hasattr(basic_model.pipeline, "feature_names_in_")
    assert not hasattr(basic_model.pipeline, "classes_")

    basic_model.train()

    assert hasattr(basic_model.pipeline, "n_features_in_")
    assert basic_model.pipeline.n_features_in_ > 0
    assert hasattr(basic_model.pipeline, "feature_names_in_")
    assert len(basic_model.pipeline.feature_names_in_) > 0
    assert hasattr(basic_model.pipeline, "classes_")
    assert hasattr(basic_model.pipeline, "_final_estimator")


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_log_model_on_databricks(logged_basic_model: BasicModel) -> None:
    """Test logging of a basic model on Databricks.

    This test verifies that the experiment for the logged basic model exists and prints its artifact location.

    :param logged_basic_model: The basic model that has been logged
    """
    experiment = mlflow.get_experiment_by_name(logged_basic_model.experiment_name)
    assert experiment is not None
    print(f"{experiment.artifact_location} = ")
    assert experiment.artifact_location


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_register_model_on_databricks(logged_basic_model: BasicModel) -> None:
    """Test that register_model_on_databricks."""
    # experiment = mlflow.get_experiment_by_name(logged_basic_model.experiment_name) # noqa
    logged_basic_model.register_model()

    model_name = f"{logged_basic_model.catalog_name}.{logged_basic_model.schema_name}.{logged_basic_model.model_name}"  # Replace with your model's full name
    registered_models = search_registered_model_versions(full_model_name=model_name)
    assert registered_models

    if registered_models:
        print(f"Model '{model_name}' is registered.")
        for mv in registered_models:
            print(f"Name: {mv.name}")
            print(f"Version: {mv.version}")
            print(f"Stage: {mv.current_stage}")
            print(f"Description: {mv.description}")

        # delete the mess
        delete_registered_model(model_name=model_name)
    else:
        print(f"Model '{model_name}' is not registered.")


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_retrieve_current_run_dataset_on_databricks(logged_basic_model: BasicModel) -> None:
    """Test the retrieval of the current run dataset on Databricks.

    This test checks if the retrieve_current_run_dataset method returns a non-None value when executed on Databricks.

    :param logged_basic_model: An instance of BasicModel that has been logged
    """
    dataset = logged_basic_model.retrieve_current_run_dataset()
    assert dataset is not None


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_retrieve_current_run_metadata_on_databricks(logged_basic_model: BasicModel) -> None:
    """Test the retrieval of current run metadata on Databricks.

    This test verifies that the metadata retrieved from a logged basic model is not None.

    :param logged_basic_model: An instance of BasicModel that has been logged
    """
    metadata = logged_basic_model.retrieve_current_run_metadata
    assert metadata is not None


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test_load_latest_model_and_predict_on_databricks(logged_basic_model: BasicModel) -> None:
    """Test loading the latest model and making predictions on Databricks.

    This test registers the model, loads it, and makes predictions on a sample dataset.

    :param logged_basic_model: An instance of BasicModel that has been logged.
    """
    logged_basic_model.register_model()

    reg_basic_model = logged_basic_model
    table_name = f"{reg_basic_model.catalog_name}.{reg_basic_model.schema_name}.test_set"

    spark = SparkSession.builder.getOrCreate()
    input = spark.table(table_name).limit(10).drop(logged_basic_model.target).toPandas()
    predictions = reg_basic_model.load_latest_model_and_predict(input_data=input)

    print(f"{predictions = }")
    assert len(predictions) > 0
    assert len(predictions) == 10

    #  clean up the mess
    model_name = f"{logged_basic_model.catalog_name}.{logged_basic_model.schema_name}.{logged_basic_model.model_name}"  # Replace with your model's full name
    registered_models = search_registered_model_versions(full_model_name=model_name)

    if registered_models:
        delete_registered_model(model_name=model_name)
