# Databricks notebook source

!pip install /Volumes/mlops_dev/acikgozm/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
import mlflow
import os
import pathlib
from dotenv import load_dotenv

from hotel_reservations.config import Config, Tags
from hotel_reservations.utility import setup_logging
from hotel_reservations.utility import is_databricks
from hotel_reservations.basic_model import BasicModel
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
basic_model = BasicModel(config=CONFIG, tags=tags)

# COMMAND ----------

basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------

basic_model.train()

# COMMAND ----------
basic_model.log_model()


# COMMAND ----------
run_id = mlflow.search_runs(experiment_names=[CONFIG.experiment_name], filter_string="tags.branch='dev'").run_id[0]
print(f'{run_id = }')

model= mlflow.sklearn.load_model(f'runs:/{run_id}/{CONFIG.model.artifact_path}')

# COMMAND ----------
current_run_dataset=basic_model.retrieve_current_run_dataset()
current_run_dataset

# COMMAND ----------
current_run_metadata = basic_model.retrieve_current_run_metadata()
current_run_metadata

# COMMAND ----------
basic_model.register_model()

# COMMAND ----------
test_set = spark.table(f'{CONFIG.catalog_name}.{CONFIG.schema_name}.test_set').limit(10)
# COMMAND ----------
X_test = test_set.drop(CONFIG.target.alias).toPandas()
# COMMAND ----------
predictions_df = basic_model.load_latest_model_and_predict(X_test)
# COMMAND ----------
display(predictions_df)

# COMMAND ----------
# %sql
# DROP TABLE IF EXISTS mlops_dev.acikgozm.extra_set
