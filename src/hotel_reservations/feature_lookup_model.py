"""FeatureLookUp model."""

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservations.config import Config, Tags
from hotel_reservations.utility import get_delta_table_version, is_databricks

if is_databricks():
    spark = SparkSession.builder.getOrCreate()


class FeatureLookUpModel:
    """FeatureLookUpModel.

    This class implements feature engineering and model training using Databricks Feature Engineering Client.
    """

    def __init__(self, config: Config, tags: Tags) -> None:
        """Initialize FeatureLookUpModel with configurations and tags."""
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()
        self.config = config
        self.tags = tags.model_dump()

        # Initilization settings from config
        self.num_features = self.config.features.numerical
        self.cat_features = self.config.features.categorical
        self.target = self.config.target.alias
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        self.experiment_name = self.config.experiment_name
        self.model_name = self.config.model.name
        self.model_artifact_path = self.config.model.artifact_path

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_features"
        self.feature_function_name = f"{self.catalog_name}.{self.schema_name}.calculate_lead_time"

        # disable autologging in order not to have multiple experiments and runs :)
        mlflow.autolog(disable=True)

    def create_feature_table(self) -> None:
        """Create or replace the hotel_features table and populate it.

        This table stores hotel features used for model training.
        """
        spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (booking_id STRING NOT NULL, repeated_guest INT, no_of_previous_cancellations INT, no_of_previous_bookings_not_canceled INT);
        """)

        spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT booking_pk  PRIMARY KEY(booking_id);")
        spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        # We have to have all the fields from both train_set and test_set
        spark.sql(
            f""
            f"INSERT INTO {self.feature_table_name} SELECT booking_id, repeated_guest, "
            f"no_of_previous_cancellations, no_of_previous_bookings_not_canceled "
            f"FROM {self.catalog_name}.{self.schema_name}.train_set"
        )

        spark.sql(
            f""
            f"INSERT INTO {self.feature_table_name} SELECT booking_id, repeated_guest, "
            f"no_of_previous_cancellations, no_of_previous_bookings_not_canceled "
            f"FROM {self.catalog_name}.{self.schema_name}.test_set"
        )

        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the lead_time.

        This function calculates the difference in days between the date of arrival and the date of booking.
        """
        spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.feature_function_name}(date_of_arrival DATE, date_of_booking DATE)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return (date_of_arrival-date_of_booking).days
        $$
        """)

        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        It also prepares the data by dropping unnecessary columns.
        """
        if not is_databricks():
            raise ValueError("This function is only supported on Databricks.")

        logger.info("Loading data from Databricks tables...")
        drop_list = [
            "lead_time",
            "repeated_guest",
            "no_of_previous_cancellations",
            "no_of_previous_bookings_not_canceled",
        ]

        # self.train_set_spark = spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(*drop_list)
        self.train_set_spark = spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        features = [self.num_features + self.cat_features]
        self.train_set_spark = self.train_set_spark.select(*features).drop(*drop_list)
        # self.train_set = self.train_set_spark.toPandas()  # noqa
        self.train_set = self.train_set_spark
        self.test_set = spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        data_version = get_delta_table_version(self.catalog_name, self.schema_name, "train_set")
        self.data_version = str(data_version) if is_databricks() else "0"

        logger.info(f"✅ Data successfully loaded by dropping {', '.join(drop_list)}.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        It uses Feature Engineering Client to create a training set by joining features.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=[
                        "repeated_guest",
                        "no_of_previous_cancellations",
                        "no_of_previous_bookings_not_canceled",
                    ],
                    lookup_key="booking_id",
                ),
                FeatureFunction(
                    udf_name=self.feature_function_name,
                    output_name="lead_time",
                    input_bindings={
                        "date_of_arrival": "date_of_arrival",
                        "date_of_booking": "date_of_booking",
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc", "market_segment_type"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["lead_time"] = (self.test_set["date_of_arrival"] - self.test_set["date_of_booking"]).apply(
            lambda x: x.days
        )

        # self.X_train = self.training_df[self.num_features + self.cat_features + ["lead_time"]]  # noqa
        self.X_train = self.training_df[self.num_features + self.cat_features]
        self.y_train = self.training_df[self.target]

        # self.X_test = self.test_set[self.num_features + self.cat_features + ["lead_time"]]  # noqa
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train_log_model(self) -> None:
        """Train the model and log results to MLflow.

        This includes preprocessing, model training, and metric evaluation.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LGBMClassifier(**self.parameters)),  # Replace with your model
            ]
        )
        logger.info("Preprocessing pipeline defined.")

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            logger.info(f"Current run_id: {self.run_id}")
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            # Evaulate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_test = roc_auc_score(self.y_test, y_pred)
            clf_report = classification_report(self.y_test, y_pred)

            logger.info(f"Accuracy Report: {accuracy}")
            logger.info(f"AUC      Report: {auc_test}")
            logger.info(f"Classification Report: \n{clf_report}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LGBMClassifier with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc", auc_test)

            # log model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path=self.model_artifact_path,
                training_set=self.training_set,
                signature=signature,
                # input_example=self.X_train.head(4),  #  noqa  test and see
            )

        logger.info(f"✅ Model logged successfully to {self.model_artifact_path}.")

    def register_model(self) -> None:
        """Register the model in UC and set the latest version alias.

        This method registers the LightGBM pipeline model in MLflow, logs the registered version,
        and sets the 'latest-model' alias to the newly registered version.
        """
        logger.info("Registering the model in UC")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/{self.model_artifact_path}",  # lightgbm-pipeline-model",
            name=f"{self.catalog_name}.{self.schema_name}.{self.model_name}",  # hotel_reservations_model_basic",
            tags=self.tags,
        )
        logger.info(f"✅ Model '{registered_model.name}' registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=registered_model.name,
            alias="latest-model",
            version=latest_version,
        )
        logger.info("The model is registered in UC")

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        :param X: Input DataFrame for prediction.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.{self.model_name}@latest-model"
        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
