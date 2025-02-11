"""FeatureLookUp model."""

import numpy as np
import pandas as pd
from databricks import feature_engineering
from databricks.sdk import Workspace
from loguru import logger
from pyspark.sql import SparkSession

from hotel_reservations.config import Config, Tags
from hotel_reservations.utility import get_delta_table_version, is_databricks

if is_databricks():
    spark = SparkSession.builder.getOrCreate()


class FeatureLookUpModel:
    """FeatureLookUpModel."""

    def __init__(self, config: Config, tags: Tags) -> None:
        self.workspace = Workspace()
        self.fe = feature_engineering.FeatureEngineeringClient()
        self.config = config
        self.tags = tags.model_dump()

        # Initilization settings from config
        self.experiment_name = self.config.experiment_name
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target.alias
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_features"
        self.feature_function_name = f"{self.catalog_name}.{self.schema_name}.calculate_lead_time"

    def load_data(self) -> None:
        """Load training and testing data from Delta tables."""
        if not is_databricks():
            raise ValueError("This function is only supported on Databricks.")

        logger.info("Loading data from Databricks tables...")
        data_version = get_delta_table_version(self.catalog_name, self.schema_name, "train_set")
        drop_list = [
            "lead_time",
            "repeated_guest",
            "no_of_previous_cancellations",
            "no_of_previous_bookings_not_canceled",
        ]

        self.train_set_spark = spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(*drop_list)
        self.train_set = self.train_set_spark.toPandas()

        self.test_set = spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = str(data_version) if is_databricks() else "0"

        logger.info("✅ Data successfully loaded by dropping {', '.join(drop_list)}.")

    def create_feature_table(self) -> None:
        """Create or replace the hotel_features table and populate it."""
        # self.fe.create_table(name=self.feature_table_name, primary_key=["booking_id"], df=self.train_set)  # noqa

        spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {self.feature_table_name}
        (booking_id STRING NOT NULL, repeated_guest INT, no_of_previous_cancellations INT, no_of_previous_bookings_not_canceled INT);
        """)

        spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT booking_pk  PRIMARY KEY(booking_id);")
        spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        # We have to have all the fields from both train_set and test_set
        spark.sql(
            f""
            f"INSERT INTO {self.feature_table_name} SELECT biiking_id, repeated_guest, "
            f"no_of_previous_cancellations, no_of_previous_bookings_not_canceled "
            f"FROM {self.catalog_name}.{self.schema_name}.train_set"
        )

        spark.sql(
            f""
            f"INSERT INTO {self.feature_table_name} SELECT biiking_id, repeated_guest, "
            f"no_of_previous_cancellations, no_of_previous_bookings_not_canceled "
            f"FROM {self.catalog_name}.{self.schema_name}.test_set"
        )

        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the lead_time."""
        spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.feature_function_name}(date_of_arrival TIMESTAMP, date_of_booking TIMESTAMP)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return (date_of_arrival-date_of_booking).days
        $$""")
        logger.info("✅ Feature function defined.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables."""
        pass

    def train_log_model(self) -> None:
        """Train the model and log results to MLflow."""
        pass

    def register_model(self) -> None:
        """Register the model."""
        pass

    def load_latest_model_and_predict(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions."""
        pass
