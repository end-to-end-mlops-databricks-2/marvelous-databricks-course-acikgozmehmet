# Databricks notebook source

!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl
# COMMAND ----------

%restart_python

# COMMAND ----------
from hotel_reservations import __version__
print(__version__)

import os
import pathlib
from dotenv import load_dotenv
from hotel_reservations import __version__
from hotel_reservations.config import Config
from hotel_reservations.data_ingestion import DataLoader
from hotel_reservations.utility import setup_logging
from hotel_reservations.utility import is_databricks

print(__version__)

# COMMAND ----------
envfile_path=pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f'{envfile_path =}')

# COMMAND ----------
load_dotenv(envfile_path)

INGESTION_LOGS = os.environ['INGESTION_LOGS']
INGESTION_LOGS = pathlib.Path(INGESTION_LOGS).resolve().as_posix()
print(f"{INGESTION_LOGS = }")

# COMMAND ----------

setup_logging(INGESTION_LOGS)

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
extra_set = dataloader.split_and_extract_data()

# COMMAND ----------

train_set, test_set = dataloader.split_data()

# COMMAND ----------

dataloader.save_to_catalog(train_set=train_set, test_set=test_set, extra_set=extra_set)

# COMMAND ----------

# COMMAND ----------

# COMMAND ----------
# %sql
# DROP TABLE IF EXISTS mlops_dev.acikgozm.extra_set
