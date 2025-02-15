# Databricks notebook source
from IPython.core.display_functions import display

!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

%restart_python

# COMMAND ----------
import os
import pathlib


import requests
from dotenv import load_dotenv

import mlflow
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from hotel_reservations.config import Config, Tags
from hotel_reservations.utility import setup_logging, call_endpoint
from hotel_reservations.utility import is_databricks
from hotel_reservations.serving import ModelServing
from hotel_reservations import __version__

print(__version__)

# COMMAND ----------
# spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# COMMAND ----------
os.environ['DBR_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DBR_HOST'] = spark.conf.get('spark.databricks.workspaceUrl')
# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------
envfile_path=pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f'{envfile_path =}')

# COMMAND ----------
load_dotenv(envfile_path)

DEPLOYMENT_LOGS = os.environ['DEPLOYMENT_LOGS']
DEPLOYMENT_LOGS = pathlib.Path(DEPLOYMENT_LOGS).resolve().as_posix()
print(f"{DEPLOYMENT_LOGS = }")

# COMMAND ----------

setup_logging(DEPLOYMENT_LOGS)

# COMMAND ----------
if is_databricks():
    CONFIG_FILE_PATH = pathlib.Path("../project_config.yml").resolve().as_posix()


print(f"{CONFIG_FILE_PATH = }")

# COMMAND ----------
CONFIG = Config.from_yaml(CONFIG_FILE_PATH)
tags = Tags(branch="dev")


# COMMAND ----------
# configuration changed for testing
# configuration changed for testing
CONFIG.experiment_name = CONFIG.experiment_name + "-basic"
CONFIG.model.name = CONFIG.model.name + "_basic"
tags = Tags(branch="dev")

logger.info(f"{CONFIG.experiment_name = }")
logger.info(f"{CONFIG.model.name = }")
logger.info(f"{CONFIG.model.artifact_path = }")

# COMMAND ----------
# Initialize model
full_model_name=f"{CONFIG.catalog_name}.{CONFIG.schema_name}.{CONFIG.model.name}"
print(f"{full_model_name = }")

endpoint_name = CONFIG.model.name.replace("_", "-") +"-serving"
print(f"{endpoint_name = }")

# COMMAND ----------
model_serving = ModelServing(model_name=full_model_name, endpoint_name=endpoint_name)

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------
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
print(dataframe_records)
# COMMAND ----------
status_code, response_text = call_endpoint(endpoint_name, dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")
# COMMAND ----------

