"""Script for refreshing monitoring fe_model."""

import os
import pathlib

from dotenv import load_dotenv
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import Config, Tags
from hotel_reservations.feature_lookup_model import FeatureLookUpModel
from hotel_reservations.serving import FeatureLookupServing
from hotel_reservations.supervision import MonitoringManager
from hotel_reservations.utility import create_parser, setup_logging

# Initialize Spark session and Databricks utilities
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Load environment variables
envfile_path = pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f"{envfile_path =}")

load_dotenv(envfile_path)

# Set up logging
MONITORING_LOGS = os.environ["MONITORING_LOGS"]
MONITORING_LOGS = pathlib.Path(MONITORING_LOGS).resolve().as_posix()
print(f"{MONITORING_LOGS = }")

setup_logging(MONITORING_LOGS)

# Parse command-line arguments
args = create_parser()
logger.info(f"Root path: {args.root_path}")
logger.info(f"Environment: {args.env}")

# Load project configuration
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = Config.from_yaml(config_file=config_path, env=args.env)
tags = Tags(branch="dev")

# update the configuration with custom configuration
config.experiment_name = config.experiment_name + "-fe"
config.model.name = config.model.name + "_fe"
config.model.artifact_path = "fe-model"

logger.info(f"{config.experiment_name = }")
logger.info(f"{config.model.artifact_path = }")
logger.info(f"Updated model name: {config.model.name}")

catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = config.model.name
full_model_name = f"{catalog_name}.{schema_name}.{model_name}"
logger.info(f"Full model name: {full_model_name}")

feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
logger.info(f"Feature table name: {feature_table_name}")

endpoint_name = model_name.replace("_", "-") + "-serving"
endpoint_name = f"{endpoint_name}-{args.env}"
logger.info(f"Endpoint name: {endpoint_name}")


# Instantiate FeatureLookUpModel to update the feature table
fe_model = FeatureLookUpModel(config=config, tags=tags)
logger.info("An instance of FeatureLookUpModel is instantiated.")

# Instantiate FeatureLookUpServing to update the online feature table
feature_model_server = FeatureLookupServing(
    endpoint_name=endpoint_name,
    model_name=full_model_name,
    feature_table_name=feature_table_name,
)

# Instantiate MonitorManager to create monitoring
inference_table_fullname = f"{catalog_name}.{schema_name}.`hotel-reservations-model-fe-serving-dev_payload`"
monitor = MonitoringManager(
    model=fe_model, serving=feature_model_server, inference_table_fullname=inference_table_fullname
)
logger.info("An instance of MonitoringManager is instantiated")

# We need to send the paths of tables to merge the ground truth with the inference table for monitoring.
# We assume to have ground truth from these tables after some time in real life.
test_set_table_fullname = f"{catalog_name}.{schema_name}.test_set"
inference_set_table_fullname = f"{catalog_name}.{schema_name}.inference_set"
# COMMAND ----------
monitor.create_or_refresh_monitoring(
    test_set_table_fullname=test_set_table_fullname, inference_set_table_fullname=inference_set_table_fullname
)
