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
from hotel_reservations.custom_model import CustomModel, load_model
from hotel_reservations.feature_lookup_model import FeatureLookUpModel
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
# configuration changed for testing
CONFIG.experiment_name = CONFIG.experiment_name + "-fe"
CONFIG.model.name = CONFIG.model.name + "_fe"
CONFIG.model.artifact_path='fe-model'

logger.info(f"{CONFIG.experiment_name = }")
logger.info(f"{CONFIG.model.name = }")
logger.info(f"{CONFIG.model.artifact_path = }")

# COMMAND ----------
%sql
DROP TABLE IF EXISTS mlops_dev.acikgozm.hotel_features

# COMMAND ----------

# COMMAND ----------
fe_model = FeatureLookUpModel(config=CONFIG, tags=tags)
# COMMAND ----------

# COMMAND ----------
fe_model.create_feature_table()

# COMMAND ----------
fe_model.define_feature_function()

# COMMAND ----------
fe_model.load_data()

# COMMAND ----------
fe_model.feature_engineering()
# COMMAND ----------
fe_model.train_log_model()
# COMMAND ----------
fe_model.register_model()
# COMMAND ----------

# COMMAND ----------
# predictions
# input = pd.read_csv(
#     (CURR_DIR / ".." / "tests" / "test_data" / "train_test_pred" / "xtest.csv").resolve().as_posix()
# ).head(10)

# COMMAND ----------
# predictions = custom_model.load_latest_model_and_predict(input_data=input)
# display(predictions)
