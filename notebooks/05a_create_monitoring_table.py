# Databricks notebook source
import itertools
!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------
%restart_python

# COMMAND ----------
import os
import pathlib

import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from loguru import logger

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations import __version__
from hotel_reservations.config import Config, Tags
from hotel_reservations.feature_lookup_model import FeatureLookUpModel
from hotel_reservations.serving import FeatureLookupServing
from hotel_reservations.monitoring import Monitor
from hotel_reservations.utility import create_parser, is_databricks, setup_logging

print(__version__)
# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)


# COMMAND ----------

# COMMAND ----------
envfile_path=pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f'{envfile_path =}')

load_dotenv(envfile_path)

MONITORING_LOGS = os.environ['MONITORING_LOGS']
MONITORING_LOGS = pathlib.Path(MONITORING_LOGS).resolve().as_posix()
print(f"{MONITORING_LOGS = }")

setup_logging(MONITORING_LOGS)

# COMMAND ----------
if is_databricks():
    CONFIG_FILE_PATH = pathlib.Path("../project_config.yml").resolve().as_posix()

print(f"{CONFIG_FILE_PATH = }")

# COMMAND ----------
config = Config.from_yaml(CONFIG_FILE_PATH)
tags = Tags(branch="dev")

# COMMAND ----------
config.experiment_name = config.experiment_name + "-fe"
config.model.name = config.model.name + "_fe"
config.model.artifact_path='fe-model'
# COMMAND ----------
logger.info(f"{config.experiment_name = }")
logger.info(f"{config.model.artifact_path = }")
logger.info(f"Updated model name: {config.model.name}")

catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = config.model.name
full_model_name = f"{catalog_name}.{schema_name}.{model_name}"
logger.info(f"Full model name: {full_model_name}")

#  I may even not need this, since it is already in fe_model.feature_table_name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
logger.info(f"Feature table name: {feature_table_name}")


env="dev"
endpoint_name = model_name.replace("_", "-") + "-serving"
endpoint_name = f"{endpoint_name}-{env}"
logger.info(f"Endpoint name: {endpoint_name}")


# COMMAND ----------
fe_model =FeatureLookUpModel(config=config, tags=tags)
logger.info("Model initiated")

# COMMAND ----------

# COMMAND ----------
# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    endpoint_name=endpoint_name,
    model_name=full_model_name,
    feature_table_name=feature_table_name,
)

# COMMAND ----------
monitor = Monitor(model=fe_model, serving= feature_model_server)
logger.info("monitor instance created")

# COMMAND ----------
# Create an inference_set table from extra_set by filtering market_segment_type='Aviation';
spark.sql(
    f"CREATE TABLE {catalog_name}.{schema_name}.inference_set "
    f"LIKE {catalog_name}.{schema_name}.extra_set;"
)

spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.inference_set "
          "ADD COLUMN update_timestamp_utc TIMESTAMP;")

# COMMAND ----------
time_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
spark.sql(
    f"INSERT INTO {catalog_name}.{schema_name}.inference_set "
    f"SELECT *, '{time_now}' as update_timestamp_utc "
    f"FROM {catalog_name}.{schema_name}.extra_set where "
    "market_segment_type='Aviation';"
)
# COMMAND ----------
# update the feature_table
fe_model.update_feature_table(tables=['inference_set'])

# COMMAND ----------
# update the online feature table
feature_model_server.update_online_table(config)

# COMMAND ----------
# Load test set from Delta table
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
test_set.head()


# COMMAND ----------
# Load inference_set from Delta table
inference_set = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_set").toPandas()
inference_set.head()

# COMMAND ----------
# required columns for inference
required_columns = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "avg_price_per_room",
    "no_of_special_requests",
    "type_of_meal_plan",
    "room_type_reserved",
    "booking_id",
    "date_of_booking",
    "date_of_arrival"
]
# COMMAND ----------
test_set_records = test_set[required_columns].to_dict(orient="records")
inference_set_records = inference_set[required_columns].to_dict(orient="records")

# COMMAND ----------
# COMMAND ----------
# start shooting with test_set
end_time = datetime.now() + timedelta(minutes=20)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = monitor.query_request(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)


# COMMAND ----------
fe_model.update_feature_table()
logger.info("Feature table updated.")


# update online table
feature_model_server.update_online_table(config)


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
# Load test set from Delta table
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(100)
# Drop feature lookup columns 
test_set = test_set.drop("lead_time", "repeated_guest", "no_of_previous_cancellations","no_of_previous_bookings_not_canceled")
test_set.head()

# COMMAND ----------
should_register_new_model=fe_model.should_register_new_model(test_set=test_set)
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
