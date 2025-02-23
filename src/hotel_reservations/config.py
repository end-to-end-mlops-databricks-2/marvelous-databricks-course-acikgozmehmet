"""Configuration schema for the credit default application."""

from __future__ import annotations

from typing import Any, Literal

import yaml
from loguru import logger
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from hotel_reservations.utility import get_current_git_sha


class Tags(BaseModel):
    """Represents a tag in a version control system.

    Contains information about the git SHA and branch associated with the tag.
    """

    git_sha: str = Field(default_factory=get_current_git_sha)
    branch: str
    job_run_id: str = Field(default=None)


class Model(BaseModel):
    """Represents a model with a name and artifact path.

    This class inherits from BaseModel and defines two attributes.
    :param name: The name of the model.
    :param artifact_path: The path to the model artifact.
    """

    name: str
    artifact_path: str


class NumFeature(BaseModel):
    """A class representing a numerical feature.

    :param name: The name of the numerical feature.
    :param dtype: The data type of the numerical feature, either 'float64' or 'int64'.
    """

    name: str
    dtype: Literal["float64", "int64", "int32", "int16", "int8"]
    alias: str


class CatFeature(BaseModel):
    """A class representing a categorical feature.

    :param name: The name of the categorical feature.
    :param dtype: The data type of the feature, which is always 'object'.
    """

    name: str
    dtype: Literal["object", "category"]
    alias: str


class Target(BaseModel):
    """A class representing a target with a name, data type, and new name.

    :param name: The name of the target.
    :param dtype: The data type of the target, which can be 'float64' or 'int64'.
    :param new_name: The new name for the target.
    """

    name: str
    dtype: Literal["float64", "int64", "int32", "object", "category"]
    alias: str


class Feature(BaseModel):
    """Represents a feature with numeric and categorical attributes.

    This class inherits from BaseModel and defines two list attributes.

    :param numeric: A list of names for numerical features.
    :param categorical: A list of names for categorical features.
    """

    numerical: list[str]
    categorical: list[str]


class Config(BaseModel):
    """A class representing a configuration schema.

    :param experiment_name: The name of the experiment.
    :param catalog_name: The name of the catalog.
    :param schema_name: The name of the schema.
    :param parameters: Parameters for model training.
    :param num_features: A list of numerical features.
    :param cat_features: An optional list of categorical features.
    :param target: The target feature.
    :param features: The features.
    :param model: The model.
    """

    experiment_name: str
    catalog_name: str
    schema_name: str
    parameters: dict[str, Any] = Field(description="Parameters for model training.")
    num_features: list[NumFeature]
    cat_features: list[CatFeature] | None = Field(default_factory=list)
    target: Target
    features: Feature
    model: Model
    # This allows the pipeline_id field to be omitted without explicitly setting it to None
    pipeline_id: str = Field(default=None)  # Optional[str] = None

    @classmethod
    def from_yaml(cls, config_file: str, env: str = "dev") -> Config:
        """Load the configuration from a specified YAML file.

        :param config_file: The path to the YAML configuration file.
        :param env: The environment to load the configuration from (default: 'dev').
        :return: An instance of Config populated with the loaded data.
        :raises FileNotFoundError: If the configuration file does not exist.
        :raises yaml.YAMLError: If there is an error parsing the YAML file.
        :raises ValidationErr: If there is a validation error in the configuration.
        """
        try:
            with open(config_file, encoding="utf-8") as file:
                config_data = yaml.safe_load(file)

            config_data["catalog_name"] = config_data[env]["catalog_name"]
            config_data["schema_name"] = config_data[env]["schema_name"]
            config_data["pipeline_id"] = config_data[env]["pipeline_id"]

            config = Config(**config_data)
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            raise
        except yaml.YAMLError as err:
            logger.error(f"Error parsing YAML configuration file: {str(err)}")
            raise
        except ValidationError as err:
            logger.error(f"Validation error in configuration: {err}")
            raise
