# Databricks notebook source
!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

%restart_python

# COMMAND ----------
from IPython.core.display_functions import display

# COMMAND ----------
import os
import pathlib

import pandas as pd
import requests
from dotenv import load_dotenv
from databricks import feature_engineering
import mlflow
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from hotel_reservations.config import Config, Tags
from hotel_reservations.utility import setup_logging, call_endpoint
from hotel_reservations.utility import is_databricks
from hotel_reservations.serving import ModelServing, FeatureServing
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

fe = feature_engineering.FeatureEngineeringClient()


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
catalog_name = CONFIG.catalog_name
schema_name = CONFIG.schema_name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_reservations_preds"
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"

# COMMAND ----------
# Initialize model
# full_model_name=f"{CONFIG.catalog_name}.{CONFIG.schema_name}.{CONFIG.model.name}"
# print(f"{full_model_name = }")

endpoint_name = CONFIG.model.name.replace("_", "-") +"-feature-serving"
print(f"{endpoint_name = }")

# COMMAND ----------
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
df = pd.concat([train_set, test_set])

model = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.hotel_reservations_model_basic@latest-model")


# COMMAND ----------
preds_df = df[["booking_id"]]
preds_df["Predicted_Default"] = model.predict(df[CONFIG.features.categorical + CONFIG.features.numerical])
preds_df = spark.createDataFrame(preds_df)


# COMMAND ----------
fe.create_table(
  name = feature_table_name,
  primary_keys=["booking_id"],
  df = preds_df,
  description = "Hotel booking cancellation predictions"
)


# COMMAND ----------
spark.sql(f"ALTER TABLE {feature_table_name} "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# COMMAND ----------
feature_serving = FeatureServing(
    feature_table_name=feature_table_name, feature_spec_name=feature_spec_name, endpoint_name=endpoint_name
)
# COMMAND ----------
# Create online table
feature_serving.create_online_table()

# COMMAND ----------
# Create feature spec
feature_serving.create_feature_spec()

# COMMAND ----------
# Deploy feature serving endpoint
feature_serving.deploy_or_update_serving_endpoint_with_retry()

# COMMAND ----------

# COMMAND ----------
dataframe_records = [{"booking_id": "INN32499"}]
print(dataframe_records)
# COMMAND ----------
status_code, response_text = call_endpoint(endpoint_name, dataframe_records)
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# delete feature spec
# from databricks.feature_engineering import FeatureEngineeringClient

# fe_client = FeatureEngineeringClient()
# fe_client.delete_feature_spec(name="mlops_dev.acikgozm.return_predictions")
