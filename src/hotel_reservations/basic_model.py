"""Basic Model implementation."""

import mlflow
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.hotel_reservations.config import Config, Tag
from src.hotel_reservations.utility import is_databricks

if is_databricks():
    spark = SparkSession.builder.getOrCreate()


class BasicModel:
    """A basic model class.

    :param name: The name of the model.
    """

    def __init__(self, config: Config, tag: Tag) -> None:
        self.config = config
        self.tags = tag.model_dump()

        # Initilization settings from config
        self.num_features = self.config.features.numerical  # [feat.alias for feat in self.config.num_features]
        self.cat_features = self.config.features.categorical  # [feat.alias for feat in self.config.cat_features]
        self.target = self.config.target.alias
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name

    def load_data(self) -> None:
        """Load training and test data from Databricks tables.

        This function loads data from Spark tables, converts them to pandas DataFrames,
        and prepares the feature and target variables for both training and test sets.

        :raises ValueError: If not running on Databricks.
        """
        if not is_databricks():
            raise ValueError("This function is only supported on Databricks.")

        logger.info("Loading data from Databricks tables...")
        self.train_set_spark = spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0"

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        logger.info(
            f"Data loaded successfully. Training set size: {self.X_train.shape} and test set size: {self.X_test.shape}."
        )

    def prepare_features(self) -> None:
        """Define the preprocessing pipeline and classifier.

        This method sets up a ColumnTransformer for categorical features and creates a Pipeline
        with the preprocessor and an LGBMClassifier.
        """
        logger.info("Defining preprocessing pipeline...")

        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)],
            remainder="passthrough",
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", LGBMClassifier(**self.parameters)),  # Replace with your model
            ]
        )
        logger.info("Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model using the pipeline and training data.

        This method fits the pipeline to the training data (X_train and y_train) and logs the process.
        """
        logger.info("Started training...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("Model trained successfully.")

    def log_model(self) -> None:
        """Log the model, its parameters, and evaluation metrics using MLflow.

        This method sets up an MLflow experiment, evaluates the model, and logs relevant information.
        """
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

            # Evaulate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            clf_report = classification_report(self.y_test, y_pred)

            logger.info(f"Accuracy Report: {accuracy}")
            logger.info(f"Classification Report: \n{clf_report}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LGBMClassifier with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)

            # log model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")

            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lightgbm-pipeline-model",
                signature=signature,
            )
        logger.info("Model logged successfully.")

    def register_model(self) -> None:  # noqa
        ...

    def retrieve_current_run_dataset(self) -> None:  # noqa
        ...

    def retrieve_current_run_metadata(self) -> None:  # noqa
        ...

    def load_latest_model_and_predict(self) -> None:  # noqa
        ...
