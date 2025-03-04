# Databricks notebook source
from pandas.core.interchange.dataframe_protocol import DataFrame
!pip install hotel_reservations-latest-py3-none-any.whl
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
from pyspark.sql import SparkSession
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

#  I may even not need this, since it is already in fe_model.feature_table_name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_features"
logger.info(f"Feature table name: {feature_table_name}")


env="dev"
endpoint_name = model_name.replace("_", "-") + "-serving"
endpoint_name = f"{endpoint_name}-{env}"
logger.info(f"Endpoint name: {endpoint_name}")

# COMMAND ----------
# Instantiate FeatureLookUpModel to update the feature table
fe_model =FeatureLookUpModel(config=config, tags=tags)
logger.info("Model initiated")

# COMMAND ----------

# COMMAND ----------
# Instantiate FeatureLookUpServing to update the online feature table
feature_model_server = FeatureLookupServing(
    endpoint_name=endpoint_name,
    model_name=full_model_name,
    feature_table_name=feature_table_name,
)

# COMMAND ----------
# Instantiate MonitorManager to create monitoring
monitor = MonitoringManager(model=fe_model, serving= feature_model_server)
logger.info("monitor instance created")

# COMMAND ----------
spark.table(f"{catalog_name}.{schema_name}.extra_set").select('market_segment_type').toPandas().value_counts()
# COMMAND ----------
# Create an inference_set table from extra_set by filtering market_segment_type='Aviation';

# inference_set=spark.table(f"{catalog_name}.{schema_name}.extra_set").drop("update_timestamp_utc").toPandas().query("market_segment_type=='Aviation'")
# inference_set_with_timestamp=spark.createDataFrame(inference_set).withColumn("update_timestamp_utc", F.to_utc_timestamp(F.current_timestamp(), "UTC") )
# inference_set_with_timestamp.write.mode("overwrite").saveAsTable(f"{config.catalog_name}.{config.schema_name}.inference_set")
# spark.sql(f"ALTER TABLE {config.catalog_name}.{config.schema_name}.inference_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
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
# Create the inference set
create_inference_set(market_segment_type="Aviation", catalog_name=config.catalog_name, schema_name=config.schema_name)

# COMMAND ----------
# Update the feature_table with the new inference set
fe_model.update_feature_table(tables=['inference_set'])

# COMMAND ----------
# Update the online feature table
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
# json_string = json.dumps(test_set_records, default=str)
# dataframe_records=json.loads(json_string)
# display(dataframe_records)

# COMMAND ----------
# json_string = json.dumps(dataframe_records, default=str)
# test_set_records=json.loads(json_string)

# COMMAND ----------
os.environ['DBR_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DBR_HOST'] = spark.conf.get('spark.databricks.workspaceUrl')

# COMMAND ----------
_, test_set_records = dict_to_json_to_dict(input_dict=test_set_records)
display(test_set_records)
# response = monitor.query_request(test_set_records[0])

# COMMAND ----------
_, inference_set_records = dict_to_json_to_dict(input_dict=inference_set_records)
display(inference_set_records)

# COMMAND ----------
# start shooting with inference set records
end_time = datetime.now() + timedelta(minutes=20)
for index, record in enumerate(itertools.cycle(inference_set_records)):
    if datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    status_code, response_text = call_endpoint(endpoint_name=endpoint_name, records=record)
    print(f"Response status: {status_code}")
    print(f"Response text: {response_text}")
    time.sleep(0.2)

# COMMAND ----------
end_time = datetime.now() + timedelta(minutes=30)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    status_code, response_text = call_endpoint(endpoint_name=endpoint_name, records=record)
    print(f"Response status: {status_code}")
    print(f"Response text: {response_text}")
    time.sleep(0.2)

# COMMAND ----------
monitor.create_or_refresh_monitoring()