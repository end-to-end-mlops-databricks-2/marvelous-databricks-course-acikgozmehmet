# Databricks notebook source
!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------
import os
import pathlib

from dotenv import load_dotenv
from loguru import logger

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations import __version__
from hotel_reservations.config import Config, Tags
from hotel_reservations.feature_lookup_model import FeatureLookUpModel
from hotel_reservations.utility import create_parser, is_databricks, setup_logging

print(__version__)
# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)


# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------
envfile_path=pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f'{envfile_path =}')
load_dotenv(envfile_path)

INGESTION_LOGS = os.environ['INGESTION_LOGS']
INGESTION_LOGS = pathlib.Path(INGESTION_LOGS).resolve().as_posix()
print(f"{INGESTION_LOGS = }")
setup_logging(INGESTION_LOGS)

# COMMAND ----------
if is_databricks():
    DATABRICKS_FILE_PATH = os.environ["DATA_FILEPATH_DATABRICKS"]
    CONFIG_FILE_PATH = pathlib.Path("../project_config.yml").resolve().as_posix()


print(f"{DATABRICKS_FILE_PATH = }")
print(f"{CONFIG_FILE_PATH = }")

# COMMAND ----------
config = Config.from_yaml(CONFIG_FILE_PATH)

# COMMAND ----------
# input = ["model_train_register","--root_path","/path/to/root","--env","dev","--git_sha","abc123","--job_run_id","12345","--branch", "main"]
# args = create_parser(input)

tags = Tags(branch="dev")
# COMMAND ----------
fe_model =FeatureLookUpModel(config=config, tags=tags)
logger.info("Model initiated")

# COMMAND ----------
fe_model.update_feature_table()
logger.info("Feature table updated.")

# COMMAND ----------
# Load data
fe_model.load_data()
logger.info("Data loaded.")

# COMMAND ----------
# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------
# Train the model
fe_model.train_log_model()
logger.info("Model training completed.")

# COMMAND ----------
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)
# Drop feature lookup columns and target
test_set = test_set.drop("OverallQual", "GrLivArea", "GarageCars")

# Load test set from Delta table
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)
# Drop feature lookup columns and target
X_test = test_set.drop("lead_time", "repeated_guest", "no_of_previous_cancellations","no_of_previous_bookings_not_canceled", config.target.alias)
X_test.head()

# COMMAND ----------
should_register_new_model=fe_model.should_register_new_model(test_set=X_test)
logger.info(f"{should_register_new_model = }")
# COMMAND ----------

if should_register_new_model:
    # Register new model
    latest_version = fe_model.register_model()
    logger.info(f"New model registered with version {latest_version}")
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)

# COMMAND ----------
