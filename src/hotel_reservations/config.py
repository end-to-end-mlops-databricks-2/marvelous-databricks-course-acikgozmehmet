"""Configuration schema for the credit default application."""

from __future__ import annotations

from typing import Literal

import yaml
from loguru import logger
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)


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


class Config(BaseModel):
    """A class representing a configuration schema.

    :param schema_name: The name of the schema.
    :param num_features: A list of numerical features.
    :param cat_features: An optional list of categorical features.
    :param target: The target feature.
    """

    catalog_name: str
    schema_name: str
    num_features: list[NumFeature]
    cat_features: list[CatFeature] | None = Field(default_factory=list)
    target: Target

    @classmethod
    def from_yaml(cls, config_file: str) -> Config:
        """Load the configuration from a specified YAML file.

        :param config_file: The path to the YAML configuration file.
        :return: An instance of Config populated with the loaded data.
        :raises FileNotFoundError: If the configuration file does not exist.
        :raises yaml.YAMLError: If there is an error parsing the YAML file.
        :raises ValidationErr: If there is a validation error in the configuration.
        """
        try:
            with open(config_file, encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
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
