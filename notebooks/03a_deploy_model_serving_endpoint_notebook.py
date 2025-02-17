# Databricks notebook source
!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------
%restart_python

# COMMAND ----------
import os
import pathlib
from IPython.core.display_functions import display

import mlflow
import requests
from dotenv import load_dotenv
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations import __version__
from hotel_reservations.config import Config, Tags
from hotel_reservations.serving import ModelServing
from hotel_reservations.utility import call_endpoint, is_databricks, setup_logging

logger.info(__version__)

# COMMAND ----------
# spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# COMMAND ----------
# get environment variables
os.environ['DBR_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DBR_HOST'] = spark.conf.get('spark.databricks.workspaceUrl')

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------
# Load project env
envfile_path=pathlib.Path().joinpath("../project.env").resolve().as_posix()
logger.info(f'{envfile_path =}')

load_dotenv(envfile_path)

DEPLOYMENT_LOGS = os.environ['DEPLOYMENT_LOGS']
DEPLOYMENT_LOGS = pathlib.Path(DEPLOYMENT_LOGS).resolve().as_posix()
logger.info(f"{DEPLOYMENT_LOGS = }")

# COMMAND ----------

setup_logging(DEPLOYMENT_LOGS)

# COMMAND ----------
if is_databricks():
    CONFIG_FILE_PATH = pathlib.Path("../project_config.yml").resolve().as_posix()


logger.info(f"{CONFIG_FILE_PATH = }")

# COMMAND ----------
# Load project config
CONFIG = Config.from_yaml(CONFIG_FILE_PATH)

# configuration changed for reusability
CONFIG.model.name = CONFIG.model.name + "_basic"
logger.info(f"{CONFIG.model.name = }")

full_model_name=f"{CONFIG.catalog_name}.{CONFIG.schema_name}.{CONFIG.model.name}"
logger.info(f"{full_model_name = }")

endpoint_name = CONFIG.model.name.replace("_", "-") +"-serving"
logger.info(f"{endpoint_name = }")

# COMMAND ----------
# Initialize model_serving
model_serving = ModelServing(model_name=full_model_name, endpoint_name=endpoint_name)

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint_with_retry()


# COMMAND ----------
# Let's test the endpoint

required_columns = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "type_of_meal_plan",
    "room_type_reserved"
]

# COMMAND ----------
test_set = spark.table(f"{CONFIG.catalog_name}.{CONFIG.schema_name}.test_set").toPandas()
display(test_set.head(10))

# COMMAND ----------
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
display(sampled_records)

# COMMAND ----------
dataframe_records = [[rec] for rec in sampled_records]
logger.info(dataframe_records)

# COMMAND ----------
status_code, response_text = call_endpoint(endpoint_name, dataframe_records[0])
logger.info(f"{status_code = }")
logger.info(f"{response_text = }")

# COMMAND ----------

