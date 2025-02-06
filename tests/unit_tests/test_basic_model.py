"""Unit tests for basic_models."""

import mlflow
import pytest

from hotel_reservations.basic_model import BasicModel
from hotel_reservations.config import Config, Tag
from hotel_reservations.utility import is_databricks
from tests.consts import PROJECT_DIR


def test_basic_model_init() -> None:
    """Test the initialization of the BasicModel class.

    This function creates a BasicModel instance and asserts that it is of the correct type.
    """
    config_file_path = PROJECT_DIR / "project_config.yml"
    config = Config.from_yaml(config_file_path.as_posix())

    tag = Tag(git_sha="sha1", branch="my_branch")

    basic_model = BasicModel(config=config, tag=tag)
    assert isinstance(basic_model, BasicModel)


@pytest.mark.skipif(is_databricks(), reason="Only Local")
def test_load_data_fail() -> None:
    """Test the failure case of the load_data method in BasicModel.

    This test verifies that a ValueError is raised when attempting to load data outside of Databricks.
    """
    config_file = PROJECT_DIR / "project_config.yml"
    config = Config.from_yaml(config_file)
    tag = Tag(git_sha="sha1", branch="my_branch")

    basic_model = BasicModel(config=config, tag=tag)
    with pytest.raises(ValueError) as exc:
        basic_model.load_data()
    assert "This function is only supported on Databricks." in str(exc.value)


@pytest.mark.skipif(not is_databricks(), reason="Only on Databricks")
def test_load_data() -> None:
    """Test the data loading functionality of the BasicModel.

    This function verifies that the BasicModel can successfully load data and
    that the loaded datasets have the expected properties.
    """
    config_file_path = PROJECT_DIR / "project_config.yml"
    config = Config.from_yaml(config_file_path.as_posix())

    tag = Tag(git_sha="sha1", branch="my_branch")

    basic_model = BasicModel(config=config, tag=tag)
    basic_model.load_data()
    assert basic_model.train_set.shape[0] > 1
    assert basic_model.test_set.shape[0] > 1
    assert basic_model.data_version

    assert basic_model.X_train.shape[0] > 1
    assert basic_model.X_test.shape[0] > 1
    assert basic_model.y_train.shape[0] > 1
    assert basic_model.y_test.shape[0] > 1


def test_prepare_features() -> None:
    """Test the prepare_features method in BasicModel."""
    config_file_path = PROJECT_DIR / "project_config.yml"
    config = Config.from_yaml(config_file_path.as_posix())

    tag = Tag(git_sha="sha1", branch="my_branch")

    basic_model = BasicModel(config=config, tag=tag)
    basic_model.prepare_features()

    assert basic_model.preprocessor is not None
    assert basic_model.pipeline is not None


@pytest.mark.skipif(not is_databricks(), reason="Only on Databricks")
def test_end2end_on_databricks() -> None:
    """Perform an end-to-end test of the BasicModel on Databricks.

    This test loads data, prepares features, trains the model, and logs the model using the BasicModel class.
    """
    mlflow.set_tracking_uri("databricks")
    # mlflow.set_registry_uri("databricks-uc") # noqa

    config_file_path = PROJECT_DIR / "project_config.yml"
    config = Config.from_yaml(config_file_path.as_posix())

    tag = Tag(git_sha="abcdefg12", branch="week2")

    basic_model = BasicModel(config=config, tag=tag)

    basic_model.load_data()
    basic_model.prepare_features()
    basic_model.train()
    basic_model.log_model()
