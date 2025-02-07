# Databricks notebook source

!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

%restart_python

# COMMAND ----------
from hotel_reservations import __version__
print(__version__)

import mlflow
import os
import pathlib
from dotenv import load_dotenv
from hotel_reservations import __version__
from hotel_reservations.config import Config, Tag

from hotel_reservations.utility import setup_logging
from hotel_reservations.utility import is_databricks
from hotel_reservations.basic_model import BasicModel

print(__version__)
# COMMAND ----------


# COMMAND ----------
envfile_path=pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f'{envfile_path =}')

# COMMAND ----------
load_dotenv(envfile_path)

TRAINING_LOGS = os.environ['TRAINING_LOGS']
TRAINING_LOGS = pathlib.Path(TRAINING_LOGS).resolve().as_posix()
print(f"{TRAINING_LOGS = }")

# COMMAND ----------

setup_logging(TRAINING_LOGS)

# COMMAND ----------
if is_databricks():
    # DATABRICKS_FILE_PATH = os.environ["DATA_FILEPATH_DATABRICKS"]
    CONFIG_FILE_PATH = pathlib.Path("../project_config.yml").resolve().as_posix()


# print(f"{DATABRICKS_FILE_PATH = }")
print(f"{CONFIG_FILE_PATH = }")

# COMMAND ----------
CONFIG = Config.from_yaml(CONFIG_FILE_PATH)
tags = Tag(branch="dev")

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
basic_model = BasicModel(config=CONFIG, tag=tags)


# COMMAND ----------

basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------

basic_model.train()
basic_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(experiment_names=["/Shared/hotel-reservations-basic"], filter_string="tags.branch='dev'").run_id[0]

model= mlflow.sklearn.load_model(f'runs:/{run_id}/lightgbm-pipeline-model')

# COMMAND ----------
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
basic_model.register_model()

# COMMAND ----------
test_set = spark.table(f'{CONFIG.catalog_name}.{CONFIG.schema_name}.test_set').limit(10)
# COMMAND ----------
X_test = test_set.drop(CONFIG.target.alias).toPandas()
# COMMAND ----------
predictions_df = basic_model.load_latest_model_and_predict(X_test)
# COMMAND ----------
# %sql
# DROP TABLE IF EXISTS mlops_dev.acikgozm.extra_set
