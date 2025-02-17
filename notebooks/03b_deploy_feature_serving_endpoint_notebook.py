# Databricks notebook source
!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

%restart_python

# COMMAND ----------
# Standard library imports
import os
import pathlib
import time

# Third-party library imports
import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from dotenv import load_dotenv
from IPython.core.display_functions import display
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

# Local/application-specific imports
from hotel_reservations import __version__
from hotel_reservations.config import Config, Tags
from hotel_reservations.serving import FeatureServing, ModelServing
from hotel_reservations.utility import call_endpoint, is_databricks, setup_logging

print(__version__)

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

fe = feature_engineering.FeatureEngineeringClient()


# COMMAND ----------
# Load project env
envfile_path=pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f'{envfile_path =}')

load_dotenv(envfile_path)

DEPLOYMENT_LOGS = os.environ['DEPLOYMENT_LOGS']
DEPLOYMENT_LOGS = pathlib.Path(DEPLOYMENT_LOGS).resolve().as_posix()
print(f"{DEPLOYMENT_LOGS = }")

# COMMAND ----------
# logging
setup_logging(DEPLOYMENT_LOGS)

# COMMAND ----------
if is_databricks():
    CONFIG_FILE_PATH = pathlib.Path("../project_config.yml").resolve().as_posix()

print(f"{CONFIG_FILE_PATH = }")

# COMMAND ----------
CONFIG = Config.from_yaml(CONFIG_FILE_PATH)

# COMMAND ----------
# update the configuration for reusability.
CONFIG.model.name = CONFIG.model.name + "_basic"
logger.info(f"{CONFIG.model.name = }")

# COMMAND ----------
catalog_name = CONFIG.catalog_name
schema_name = CONFIG.schema_name
model_name = CONFIG.model.name
full_model_name=f"{CONFIG.catalog_name}.{CONFIG.schema_name}.{CONFIG.model.name}"

logger.info(f"{catalog_name = }")
logger.info(f"{schema_name = }")
logger.info(f"{model_name = }")
logger.info(f"{full_model_name = }")

# COMMAND ----------
feature_table_name = f"{catalog_name}.{schema_name}.hotel_reservations_preds"
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"
endpoint_name = CONFIG.model.name.replace("_", "-") +"-feature-serving"

logger.info(f"{feature_table_name = }")
logger.info(f"{feature_spec_name = }")
logger.info(f"{endpoint_name = }")

# COMMAND ----------
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
df = pd.concat([train_set, test_set])

model = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.{model_name}@latest-model")

# COMMAND ----------
preds_df = df[["booking_id","repeated_guest","no_of_previous_cancellations","no_of_previous_bookings_not_canceled"]]
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
# Initialize feature store manager
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
feature_serving.deploy_or_update_serving_endpoint_with_retry(retry_interval=60)

# COMMAND ----------
# Let's test the endpoint
dataframe_records = [{"booking_id": "INN32499"}]
logger.info(dataframe_records)

# COMMAND ----------
start_time = time.time()
status_code, response_text = call_endpoint(endpoint_name, dataframe_records)
end_time = time.time()
execution_time = end_time - start_time

logger.info(f"{status_code = }")
logger.info(f"{response_text =}")
logger.info(f"{execution_time = } seconds")

# COMMAND ----------
serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
logger.info(f"Serving Endpoint: {serving_endpoint}")

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_split": {"columns": ["booking_id"], "data": [["INN32499"]]}},
)

# COMMAND ----------
logger.info(f"{response.status_code = }")
logger.info(f"{response.text =}")
logger.info(f"{execution_time = } seconds")

# COMMAND ----------
# Clean-up
# delete the endpoint
# feature_serving.delete_serving_endpoint()

#delete feature_spec
# feature_serving.delete_feature_spec()
