"""Script for deploying FeatureLookupServing  machine learning model."""

import os
import pathlib

from dotenv import load_dotenv
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import Config, Tags
from hotel_reservations.serving import FeatureLookupServing
from hotel_reservations.utility import create_parser, setup_logging

# Initialize Spark session and Databricks utilities
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Load environment variables
envfile_path = pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f"{envfile_path =}")

load_dotenv(envfile_path)

# Set up logging
DEPLOYMENT_LOGS = os.environ["DEPLOYMENT_LOGS"]
DEPLOYMENT_LOGS = pathlib.Path(DEPLOYMENT_LOGS).resolve().as_posix()
print(f"{DEPLOYMENT_LOGS = }")

setup_logging(DEPLOYMENT_LOGS)

# Parse command-line arguments
args = create_parser()
logger.info(f"Root path: {args.root_path}")
logger.info(f"Environment: {args.env}")

# Load project configuration
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = Config.from_yaml(config_file=config_path, env=args.env)

# Update configuration for feature engineering
config.model.name = config.model.name + "_fe"
logger.info(f"Updated model name: {config.model.name}")

catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = config.model.name
full_model_name=f"{catalog_name}.{schema_name}.{model_name}"
logger.info(f"Full model name: {full_model_name}")

feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
logger.info(f"Feature table name: {feature_table_name}")

endpoint_name = model_name.replace("_", "-") + "-serving"
endpoint_name = f"{endpoint_name}-{args.env}"
logger.info(f"Endpoint name: {endpoint_name}")


# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    endpoint_name=endpoint_name,
    model_name=full_model_name,
    feature_table_name=feature_table_name,
    version=model_version
)

# Update the online table for house features
feature_model_server.update_online_table(config=config)

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint_with_retry(retry_interval=60)
logger.info("Started deployment/update of the serving endpoint")

