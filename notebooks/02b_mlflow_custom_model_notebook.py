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
CONFIG.experiment_name = CONFIG.experiment_name + "-custom"
CONFIG.model.name = CONFIG.model.name + "_custom"
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
model_input =pd.read_csv((TRAIN_TEST_PRED_FOLDER / "xtrain.csv").resolve().as_posix())
model_output = pd.read_csv((TRAIN_TEST_PRED_FOLDER / "ypred.csv").resolve().as_posix())

# COMMAND ----------
custom_model.log_model(model_input=model_input, model_output=model_output)


# COMMAND ----------
custom_model.register_model()

# COMMAND ----------
# predictions
input = pd.read_csv(
    (CURR_DIR / ".." / "tests" / "test_data" / "train_test_pred" / "xtest.csv").resolve().as_posix()
).head(10)

# COMMAND ----------
predictions = custom_model.load_latest_model_and_predict(input_data=input)
display(predictions)
