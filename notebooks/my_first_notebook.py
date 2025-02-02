# Databricks notebook source

!pip install /Volumes/mlops_dev/acikgozm/packages/credit_default-latest-py3-none-any.whl
# COMMAND ----------

%restart_python

# COMMAND ----------
from credit_default import __version__
print(__version__)

import os
import pathlib
from dotenv import load_dotenv
from credit_default import __version__
from credit_default.config import Config
from credit_default.data_ingestion import DataLoader
from credit_default.utility import setup_logging
from credit_default.utility import is_databricks

print(__version__)

# COMMAND ----------
envfile_path=pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f'{envfile_path =}')

# COMMAND ----------
load_dotenv(envfile_path)

CLEANING_LOGS = os.environ['CLEANING_LOGS']
CLEANING_LOGS = pathlib.Path(CLEANING_LOGS).resolve().as_posix()
print(f"{CLEANING_LOGS = }")

# COMMAND ----------

setup_logging(CLEANING_LOGS)

# COMMAND ----------
if is_databricks():
    DATABRICKS_FILE_PATH = os.environ["DATA_FILEPATH_DATABRICKS"]
    CONFIG_FILE_PATH = pathlib.Path("../project_config.yml").resolve().as_posix()


print(f"{DATABRICKS_FILE_PATH = }")
print(f"{CONFIG_FILE_PATH = }")

# COMMAND ----------
CONFIG = Config.from_yaml(CONFIG_FILE_PATH)
dataloader = DataLoader(filepath=DATABRICKS_FILE_PATH, config=CONFIG)
# COMMAND ----------

dataloader.process_data()

# COMMAND ----------

train_set, test_set = dataloader.split_data()

# COMMAND ----------

dataloader.save_to_catalog(train_set=train_set, test_set=test_set)

# COMMAND ----------

# COMMAND ----------

# COMMAND ----------
# %sql
# DROP SCHEMA ml_lab_ma CASCADE
