# Databricks notebook source

# !pip install hotel_reservations-latest-py3-none-any.whl
!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl
# COMMAND ----------
%restart_python

# COMMAND ----------
import os
import pathlib
import itertools
import time
from datetime import datetime, timezone, timedelta
import json


from dotenv import load_dotenv
from loguru import logger

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from hotel_reservations import __version__
from hotel_reservations.config import Config, Tags
from hotel_reservations.feature_lookup_model import FeatureLookUpModel
from hotel_reservations.serving import FeatureLookupServing
from hotel_reservations.supervision import MonitoringManager
from hotel_reservations.utility import create_parser, is_databricks, setup_logging, dict_to_json_to_dict, call_endpoint

print(__version__)
# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
# COMMAND ----------
# Load the environment variables and set up logging
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
# Load configuration
config = Config.from_yaml(CONFIG_FILE_PATH)
tags = Tags(branch="dev")

# COMMAND ----------
# update the configuration with custom configuration
config.experiment_name = config.experiment_name + "-fe"
config.model.name = config.model.name + "_fe"
config.model.artifact_path='fe-model'
# COMMAND ----------
logger.info(f"{config.experiment_name = }")
logger.info(f"{config.model.artifact_path = }")
logger.info(f"Updated model name: {config.model.name}")

# COMMAND ----------
catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = config.model.name
full_model_name = f"{catalog_name}.{schema_name}.{model_name}"
logger.info(f"Full model name: {full_model_name}")

# COMMAND ----------
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
logger.info(f"Feature table name: {feature_table_name}")

# COMMAND ----------
endpoint_name = model_name.replace("_", "-") + "-serving"
env="dev"
endpoint_name = f"{endpoint_name}-{env}"
logger.info(f"Endpoint name: {endpoint_name}")

# COMMAND ----------
# MAGIC %md
# MAGIC #### Let's create MonitoringManager instance step by step
# COMMAND ----------
# Instantiate FeatureLookUpModel to update the feature table
fe_model = FeatureLookUpModel(config=config, tags=tags)
logger.info("An instance of FeatureLookUpModel is instantiated.")

# COMMAND ----------
# Instantiate FeatureLookUpServing to update the online feature table
feature_model_server = FeatureLookupServing(
    endpoint_name=endpoint_name,
    model_name=full_model_name,
    feature_table_name=feature_table_name,
)
logger.info("An instance of FeatureLookupServing is instantiated.")
# COMMAND ----------
# Instantiate MonitorManager to create monitoring
inference_table_fullname = f"{catalog_name}.{schema_name}.`hotel-reservations-model-fe-serving-dev_payload`"
monitor = MonitoringManager(model=fe_model, serving= feature_model_server, inference_table_fullname=inference_table_fullname)
logger.info("An instance of MonitoringManager is instantiated")

# COMMAND ----------
# MAGIC %md
# MAGIC #### Let's arrange some inference data for mimicking real life scenarios.
# COMMAND ----------
# let's see the cardinality of the market_segment_type 
spark.table(f"{catalog_name}.{schema_name}.extra_set").select('market_segment_type').toPandas().value_counts()
# COMMAND ----------
# COMMAND ----------
def create_inference_set(market_segment_type: str, catalog_name: str, schema_name:str) -> None:
    """
    Create the inference set for a given market segment type and write it to a Delta table.

    Parameters:
    - market_segment_type (str): The market segment type to filter the data (e.g., 'Aviation').
    - catalog_name (str): The catalog name where the table resides.
    - schema_name (str): The schema name where the table resides.
    """
    # Step 1: Load the extra_set table and filter by market_segment_type
    inference_set = (
        spark.table(f"{catalog_name}.{schema_name}.extra_set")
        .drop("update_timestamp_utc")
        .toPandas()
        .query(f"market_segment_type=='{market_segment_type}'")
    )

    # Step 2: Add a UTC timestamp column
    inference_set_with_timestamp = (
        spark.createDataFrame(inference_set)
        .withColumn("update_timestamp_utc", F.to_utc_timestamp(F.current_timestamp(), "UTC"))
    )

    # Step 3: Write the processed data back to a Delta table
    inference_set_with_timestamp.write.mode("overwrite").saveAsTable(
        f"{catalog_name}.{schema_name}.inference_set"
    )

    # Step 4: Enable Change Data Feed on the Delta table
    spark.sql(
        f"ALTER TABLE {catalog_name}.{schema_name}.inference_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
    )

# COMMAND ----------
# Create an inference_set table from extra_set by filtering market_segment_type='Aviation';
selected_market_segment_type = "Aviation"
create_inference_set(market_segment_type=selected_market_segment_type, catalog_name=config.catalog_name, schema_name=config.schema_name)
logger.info(f"Creating inference set with {selected_market_segment_type}")
# COMMAND ----------
# Update the feature_table with the new inference set 
fe_model.update_feature_table(tables=['inference_set'])

# COMMAND ----------
# Update the online feature table with the new inference set
feature_model_server.update_online_table(config)

# COMMAND ----------
# MAGIC %md
# MAGIC #### Let's mimic the real inference flow by using test_set and inference_set.
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
logger.info(f"test set records: \n {test_set_records[:4]}")
# COMMAND ----------
inference_set_records = inference_set[required_columns].to_dict(orient="records")
logger.info(f"inference_set_records: \n {inference_set_records[:4]}")

# COMMAND ----------
os.environ['DBR_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DBR_HOST'] = spark.conf.get('spark.databricks.workspaceUrl')

# COMMAND ----------
_, test_set_records = dict_to_json_to_dict(input_data=test_set_records)
logger.info(f"test set records: \n {test_set_records[:4]}")

# COMMAND ----------
end_time = datetime.now() + timedelta(minutes=20)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    status_code, response_text = call_endpoint(endpoint_name=endpoint_name, records=[record])
    print(f"Response status: {status_code}")
    print(f"Response text: {response_text}")
    time.sleep(0.2)

# COMMAND ----------
_, inference_set_records = dict_to_json_to_dict(input_data=inference_set_records)
logger.info(f"{inference_set_records = }")

# COMMAND ----------
# start shooting with inference set records
end_time = datetime.now() + timedelta(minutes=30)
for index, record in enumerate(itertools.cycle(inference_set_records)):
    if datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    status_code, response_text = call_endpoint(endpoint_name=endpoint_name, records=[record])
    print(f"Response status: {status_code}")
    print(f"Response text: {response_text}")
    time.sleep(0.2)

# COMMAND ----------
# We need to send the paths of tables to merge the ground truth with the inference table for monitoring. 
# We assume to have ground truth from these tables after some time in real life.
test_set_table_fullname = f"{catalog_name}.{schema_name}.test_set"
inference_set_table_fullname = f"{catalog_name}.{schema_name}.inference_set"
# COMMAND ----------
monitor.create_or_refresh_monitoring(
    test_set_table_fullname=test_set_table_fullname,
    inference_set_table_fullname=inference_set_table_fullname
)