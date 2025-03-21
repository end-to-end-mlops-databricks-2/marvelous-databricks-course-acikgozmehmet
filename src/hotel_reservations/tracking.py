"""Tracking module."""

import pytest
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking import MlflowClient


def validate_experiment_deleted(experiment: Experiment) -> None:
    """Validate that an experiment has been deleted.

    Searches for active experiments with the given name and asserts that none are found.

    :param experiment: The experiment to validate as deleted
    """
    client = MlflowClient()
    # Search for active experiments with the given name
    active_experiments = client.search_experiments(filter_string=f"name = '{experiment.name}'")
    assert not active_experiments


def assert_registered_model_deleted(model_name: str) -> None:
    """Assert that a registered model has been deleted.

    :param model_name: The name of the model to check
    """
    client = MlflowClient()
    with pytest.raises(MlflowException) as exception_context:
        client.get_registered_model(model_name)
    print(f"{exception_context.value.error_code = }")
    assert exception_context.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def delete_registered_model(model_name: str) -> None:
    """Delete a registered model from MLflow.

    This function deletes the specified model and verifies its deletion.

    :param model_name: The name of the registered model to delete.
    """
    client = MlflowClient()
    client.delete_registered_model(name=model_name)
    assert_registered_model_deleted(model_name=model_name)


def search_registered_model_versions(full_model_name: str) -> PagedList:
    """Search for registered model versions using the provided model name.

    This function uses the MlflowClient to search for model versions and prints details of each version found.

    :param full_model_name: The full name of the model to search for
    :return: A PagedList of model versions matching the search criteria
    """
    client = MlflowClient()
    model_name = full_model_name

    model_versions = client.search_model_versions(f"name='{model_name}'")
    if model_versions:
        # Search for a specific model by name
        for mv in model_versions:
            print(f"Name: {mv.name}")
            print(f"Version: {mv.version}")
            print(f"Stage: {mv.current_stage}")
            print(f"Description: {mv.description}")

    return model_versions
