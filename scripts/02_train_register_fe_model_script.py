"""Script for training and registering a machine learning model."""

import os
import pathlib

import mlflow
from dotenv import load_dotenv
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import Config, Tags
from hotel_reservations.feature_lookup_model import FeatureLookUpModel
from hotel_reservations.utility import create_parser, setup_logging

# Initialize Spark session and Databricks utilities
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Configure MLflow tracking and registry URIs
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Load environment variables
envfile_path = pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f"{envfile_path =}")

load_dotenv(envfile_path)

# Set up logging
TRAINING_LOGS = os.environ["TRAINING_LOGS"]
TRAINING_LOGS = pathlib.Path(TRAINING_LOGS).resolve().as_posix()
print(f"{TRAINING_LOGS = }")

setup_logging(TRAINING_LOGS)

# Parse command-line arguments
args = create_parser()
logger.info(f"Root path: {args.root_path}")
logger.info(f"Environment: {args.env}")

# Load configuration
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = Config.from_yaml(config_file=config_path, env=args.env)

# Update configuration for feature engineering
config.experiment_name = config.experiment_name + "-fe"
config.model.name = config.model.name + "_fe"
config.model.artifact_path = "fe-model"

logger.info(f"Experiment name: {config.experiment_name}")
logger.info(f"Model name: {config.model.name}")
logger.info(f"Model artifact path: {config.model.artifact_path}")

# Set up tags for tracking
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize the model
fe_model = FeatureLookUpModel(config=config, tags=tags)
logger.info("Model initialized")

# Update feature table
fe_model.update_feature_table()
logger.info("Feature table updated.")

# Load data
fe_model.load_data()
logger.info("Data loaded.")

# Perform feature engineering
fe_model.feature_engineering()
logger.info("Feature engineering completed")

# Train the model
fe_model.train_log_model()
logger.info("Model training completed.")

# Evaluate model
# Load test set from Delta table
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)
# Drop feature lookup columns
test_set = test_set.drop(
    "lead_time", "repeated_guest", "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"
)
test_set.head()
logger.info("Test set prepared for evaluation")

should_register_new_model = fe_model.should_register_new_model(test_set=test_set)
logger.info(f"Should register new model: {should_register_new_model}")

if should_register_new_model:
    # Register new model
    latest_version = fe_model.register_model()
    logger.info(f"New model registered with version {latest_version}")
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
else:
    logger.info("No new model registered")
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
