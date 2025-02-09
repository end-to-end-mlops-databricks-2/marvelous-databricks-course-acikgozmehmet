"""Unit tests for the config module."""

import pytest

from hotel_reservations.config import Config, Tags


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Fixture to create a temporary configuration file for testing.

    :param tmp_path_factory: A factory for creating temporary paths.
    :return: A Config object loaded from the temporary configuration file.
    """
    tmp_path = tmp_path_factory.mktemp("test_config")
    config_file = tmp_path / "config.yaml"
    config_str = """
catalog_name: my_catalog
schema_name: default
experiment_name: /Shared/my_experiment

model:
  name: hotel_reservations_model
  artifact_path: lightgbm-pipeline-model

parameters:
    random_state: 42

num_features:
  - name: ID
    dtype: int64
    alias: Id
  - name: LIMIT_BAL
    dtype: float64
    alias: Limit_bal

cat_features:

target:
  name: default.payment.next.month
  dtype: int64
  alias: Default

features:
  numerical:
  - no_of_adults
  - no_of_children
  - no_of_weekend_nights
  - no_of_week_nights


  categorical:
  - booking_id
  - type_of_meal_plan
  - room_type_reserved
  - market_segment_type

extra_field: extra_value
"""
    with open(config_file, "w", encoding="utf-8") as file:
        file.write(config_str)

    configuration = Config.from_yaml(config_file.as_posix())
    return configuration


def test_assert_config_structure(config: Config) -> None:
    """Test the structure of the configuration object.

    :param config: The configuration object to be tested.
    """
    assert isinstance(config, Config)
    assert config.catalog_name == "my_catalog"
    assert config.schema_name == "default"


def test_assert_num_features(config: Config) -> None:
    """Test to assert the number of features in the configuration.

    :param config: Configuration object containing feature information.
    """
    assert len(config.num_features) == 2
    assert config.num_features[0].name == "ID"
    assert config.num_features[0].alias == "Id"
    assert config.num_features[0].dtype == "int64"


def test_assert_cat_features(config: Config) -> None:
    """Test to assert that categorical features in the configuration are None.

    :param config: An instance of Config containing the configuration settings.
    """
    assert config.cat_features is None


def test_assert_target(config: Config) -> None:
    """Test the target configuration of the provided Config object.

    :param config: The Config object containing target configuration.
    """
    assert config.target.name == "default.payment.next.month"
    assert config.target.dtype == "int64"
    assert config.target.alias == "Default"


def test_assert_extra_field(config: Config) -> None:
    """Test to assert the configuration schema name and check for extra fields.

    :param config: The configuration object to be tested.
    """
    assert config.schema_name == "default"
    assert not hasattr(config, "extra_field")


def test_create_tag() -> None:
    """Test the creation of a tag using the Config object.

    :param config: The Config object containing target configuration.
    """
    tags = Tags(branch="test")
    assert tags.git_sha
    assert tags.branch == "test"
