# Databricks notebook source
import os
import pathlib

from dotenv import load_dotenv
import mlflow
from loguru import logger
from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score  # classification_report, confusion_matrix,
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from pyspark.sql import SparkSession
from lightgbm import LGBMClassifier

from hotel_reservations import __version__
from hotel_reservations.config import Config
from hotel_reservations.data_ingestion import DataLoader
from hotel_reservations.utility import setup_logging

print(__version__)
# COMMAND ----------
envfile_path=pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f'{envfile_path =}')

# COMMAND ----------
load_dotenv(envfile_path)
TRAINING_LOGS = os.environ['TRAINING_LOGS']
TRAINING_LOGS = pathlib.Path(TRAINING_LOGS).resolve().as_posix()
print(f'{TRAINING_LOGS = }')

# COMMAND ----------
setup_logging(TRAINING_LOGS)

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
# COMMAND ----------
CONFIG_FILE_PATH = pathlib.Path("../project_config.yml").resolve().as_posix()
CONFIG = Config.from_yaml(CONFIG_FILE_PATH)
schema_name = CONFIG.schema_name
parameters = CONFIG.parameters

# COMMAND ----------
train_set_spark = spark.table(f"{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{schema_name}.test_set").toPandas()

logger.info(f"Loaded data from  {schema_name}.train_set and {schema_name}.test_set")

X_train = train_set.drop(columns=["Default", "Id", "update_timestamp_utc"])
y_train = train_set["Default"]

X_test = test_set.drop(columns=["Default", "Id", "update_timestamp_utc"])
y_test = test_set["Default"]

logger.info(f"Train and test data loaded with shapes {X_train.shape} and {X_test.shape}")

# COMMAND ----------

features_robust = CONFIG.features.robust
print(features_robust)
# COMMAND ----------
preprocessor = ColumnTransformer(
    transformers=[("robust_scaler", RobustScaler(), features_robust)],
    remainder="passthrough",
)
logger.info(f"ColumnTransformer created with {features_robust} features")

# COMMAND ----------
# Create the pipeline with preprocessing and the LightGBM classifier
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))])
logger.info(f"Pipeline created with {parameters} parameters")

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name="/Shared/credit_default")
logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------
# Start an MLflow run to track the training process
with mlflow.start_run(tags={"branch": "serving"}) as run:
    run_id = run.info.run_id

    # Train the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    auc_test = roc_auc_score(y_test, y_pred)

    print("Test AUC:", auc_test)

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("AUC", auc_test)

    # Log the input dataset
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=y_pred)
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)

    logger.info(f"Mlflow experiment completed")

# COMMAND ----------
# COMMAND ----------
# COMMAND ----------
# COMMAND ----------
# COMMAND ----------
# COMMAND ----------
# COMMAND ----------
# COMMAND ----------
# COMMAND ----------
