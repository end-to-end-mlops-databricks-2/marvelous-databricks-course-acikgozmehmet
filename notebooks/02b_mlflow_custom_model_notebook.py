# Databricks notebook source

!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

%restart_python

# COMMAND ----------
import mlflow
import os
import pathlib
from dotenv import load_dotenv
from loguru import logger
import pandas as pd

from hotel_reservations.config import Config, Tags
from hotel_reservations.utility import setup_logging
from hotel_reservations.utility import is_databricks
from hotel_reservations.basic_model import BasicModel
from hotel_reservations.custom_model import CustomModel, load_model
from hotel_reservations import __version__

print(__version__)

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


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
    CONFIG_FILE_PATH = pathlib.Path("../project_config.yml").resolve().as_posix()


print(f"{CONFIG_FILE_PATH = }")

# COMMAND ----------
CONFIG = Config.from_yaml(CONFIG_FILE_PATH)
tags = Tags(branch="dev")


# COMMAND ----------
# some changes on the config file to multi-purpose usage
CONFIG.experiment_name = CONFIG.experiment_name.replace("-basic","-custom")
CONFIG.model.name = CONFIG.model.name.replace("_basic","_custom")
CONFIG.model.artifact_path='custom-model'

logger.info(f"{CONFIG.experiment_name = }")
logger.info(f"{CONFIG.model.name = }")
logger.info(f"{CONFIG.model.artifact_path = }")

# COMMAND ----------
# custom model file location
CURR_DIR = pathlib.Path()
custom_model_file_path =   (CURR_DIR / ".." /"tests" /"test_data" / "lightgbm-pipeline-model" / "model.pkl").resolve()
logger.info(f"{custom_model_file_path.as_posix() = }")

# COMMAND ----------
# Let's have the custom model
model = load_model(custom_model_file_path.as_posix())

# COMMAND ----------
code_paths =[]
custom_model = CustomModel(config=CONFIG, tags=tags, model=model, code_paths=code_paths)
# COMMAND ----------
# We need to provide model_input and model_output to infer_schema
TRAIN_TEST_PRED_FOLDER =   CURR_DIR / ".." /"tests" /"test_data" / "train_test_pred"
model_input =pd.read_csv((TRAIN_TEST_PRED_FOLDER / "xtain.csv").resolve().as_posix())
model_output = pd.read_csv((TRAIN_TEST_PRED_FOLDER / "ypred.csv").resolve().as_posix())

# COMMAND ----------
custom_model.log_model(model_input=model_input, model_output=model_output)


# COMMAND ----------
custom_model.register_model()


# run_id = mlflow.search_runs(experiment_names=[CONFIG.experiment_name], filter_string="tags.branch='dev'").run_id[0]
# print(f'{run_id = }')
#
# model= mlflow.sklearn.load_model(f'runs:/{run_id}/{CONFIG.model.artifact_path}')
#
# # COMMAND ----------
# current_run_dataset=basic_model.retrieve_current_run_dataset()
# current_run_dataset
#
# # COMMAND ----------
# current_run_metadata = basic_model.retrieve_current_run_metadata()
# current_run_metadata
#
# # COMMAND ----------
# basic_model.register_model()
#
# # COMMAND ----------
# test_set = spark.table(f'{CONFIG.catalog_name}.{CONFIG.schema_name}.test_set').limit(10)
# # COMMAND ----------
# X_test = test_set.drop(CONFIG.target.alias).toPandas()
# # COMMAND ----------
# predictions_df = basic_model.load_latest_model_and_predict(X_test)
# # COMMAND ----------
# display(predictions_df)
#
# COMMAND ----------
# %sql
# DROP TABLE IF EXISTS mlops_dev.acikgozm.extra_set
