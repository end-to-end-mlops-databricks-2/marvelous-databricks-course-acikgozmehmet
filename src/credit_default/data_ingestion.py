"""Data cleaning for credit default data using Spark."""
# pylint disable=invalid-name

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError
from pyspark.errors import AnalysisException
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split

from credit_default.config import Config
from credit_default.utility import is_databricks

# Load environment variables
load_dotenv()


# pylint disable=invalid-name
class DataLoader:
    """
    A class for cleaning and processing data.

    :param filepath: The path to the data file to be loaded.
    :param config: Configuration object for data cleaning.
    """

    def __init__(self, filepath: str, config: Config):
        """
        Initialize the DataCleaner with the given file path and configuration.

        :param filepath: The path to the data file to be loaded.
        :param config: Configuration object for data cleaning.
        """
        self.df = self._load_data(filepath)  # pylint: disable=invalid-name
        self.config = config
        self.config.cat_features = [] if self.config.cat_features is None else self.config.cat_features
        self.processed: bool = False

        logger.info(f"DataLoader initialized with {filepath} and configuration")

    def _load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a CSV file into a DataFrame.

        :param filepath: The path to the CSV file.
        :return: A DataFrame containing the loaded data.
        :raises Exception: If the DataFrame is empty or if there is an error in loading data.
        """
        try:
            logger.info(f"Loading data from {filepath}")
            if is_databricks():
                spark = SparkSession.builder.getOrCreate()
                # pylint: disable=invalid-name
                df = spark.read.csv(filepath, header=True, inferSchema=True).toPandas()
                logger.info(f"Data loaded from Databricks successfully with shape: {df.shape}")
            else:
                df = pd.read_csv(  # pylint: disable=invalid-name
                    filepath_or_buffer=filepath, header="infer"
                )  # pylint: disable=invalid-name
                logger.info(f"Data loaded from local successfully with shape: {df.shape}")
            if df.empty:
                logger.error("Dataframe is empty")
                raise Exception("Dataframe is empty")  # pylint: disable=broad-exception-raised
            return df
        except FileNotFoundError as err:
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}") from err
        except AnalysisException as err:
            logger.error(f"Path does not exist:  {str(err)}")
            raise AnalysisException("Path does not exist: {str(err)}") from err
        except pd.errors.EmptyDataError as err:
            logger.error(f"Error in loading data: {str(err)}")
            raise Exception(  # pylint: disable=broad-exception-raised
                f"Error in loading data: {str(err)}"
            ) from err

    def _validate_columns(self) -> None:
        """
        Validate the presence of required columns in the DataFrame.

        Checks if the columns specified in the configuration are present in the DataFrame.
        Raises an exception if any required columns are missing.

        :raises Exception: If any required columns are missing in the DataFrame.
        :param self: The instance of the class containing the configuration and DataFrame.
        """
        columns_to_check = (
            [feature.name for feature in self.config.num_features]
            + [feature.name for feature in self.config.cat_features]
            + [self.config.target.name]
        )

        missing_columns = [col for col in columns_to_check if col not in self.df.columns.tolist()]
        if missing_columns:
            logger.error(f"Missing columns in data: {missing_columns}")
            raise Exception(  # pylint: disable=broad-exception-raised
                f"Missing columns in data: {missing_columns})"
            )
        logger.info("All required columns are present in the data")

    def _rename_columns(self) -> None:
        """
        Rename the columns in the DataFrame based on the configuration.

        This method renames the columns in the DataFrame to match the names specified in the
        configuration for numerical, categorical features, and the target variable.
        """
        # rename columns
        for num_feat in self.config.num_features:
            self.df.rename(columns={num_feat.name: num_feat.alias}, inplace=True)

        for cat_feat in self.config.cat_features:
            self.df.rename(columns={cat_feat.name: cat_feat.alias}, inplace=True)

        self.df.rename(columns={self.config.target.name: self.config.target.alias}, inplace=True)

        logger.info("Columns renamed successfully")

    def _convert_column_types(self) -> None:
        """
        Convert column types to the DataFrame columns based on the configuration.

        This method updates the DataFrame's columns to the specified data types
        for numerical, categorical features, and the target variable.
        """
        for num_feat in self.config.num_features:
            self.df[num_feat.alias] = self.df[num_feat.alias].astype(num_feat.dtype)

        for cat_feat in self.config.cat_features:
            self.df[cat_feat.alias] = self.df[cat_feat.alias].astype(cat_feat.dtype)

        self.df[self.config.target.alias] = self.df[self.config.target.alias].astype(self.config.target.dtype)

        logger.info("Columns converted to specified data types")

    def _validate_column_types(self) -> None:
        """Validate data types of key columns."""
        for num_feat in self.config.num_features:
            if not np.issubdtype(self.df[num_feat.alias].dtype, num_feat.dtype):
                raise Exception(  # pylint: disable=broad-exception-raised
                    f"Numeric column '{num_feat.alias}' must be {num_feat.dtype}"
                )

        for cat_feat in self.config.cat_features:
            if not np.issubdtype(self.df[cat_feat.alias].dtype, cat_feat.dtype):
                # pylint: disable=broad-exception-raised
                raise Exception(f"Numeric column '{cat_feat.alias}' must be {cat_feat.dtype}")

        if not np.issubdtype(self.df[self.config.target.alias].dtype, self.config.target.dtype):
            raise Exception(  # pylint: disable=broad-exception-raised
                f"Target column '{self.config.target.alias}' must be {self.config.target.dtype}"
            )

        logger.info("Data types validation passed")

    def _validate_processed_data(self) -> None:
        """
        Validate the processed data to ensure it meets the required quality checks.

        This method checks for empty DataFrames, missing numeric and categorical columns,
        and unexpected null values. It raises exceptions with appropriate messages if any
        of the checks fail.

        :raises ValueError: If the DataFrame is empty, if required columns are missing,
        """
        # check if the dataframe is empty
        if self.df.empty:
            message = "Preprocessing resulted in an empty Dataframe"
            logger.error(message)
            raise ValueError(message)

        # check if all features  are present in columns
        columns = self.df.columns.tolist()
        num_features = [num_feat.alias for num_feat in self.config.num_features]
        cat_features = [cat_feat.alias for cat_feat in self.config.cat_features]
        if not set(num_features).issubset(set(columns)):
            message = f"Missing numneric columns {set(num_features).difference(set(columns))} after preprocessing"
            logger.info(message)
            raise ValueError(message)

        if not set(cat_features).issubset(set(columns)):
            message = f"Missing categorical columns {set(cat_features).difference(set(columns))} after preprocessing"
            logger.info(message)
            raise ValueError(message)

        if not set([self.config.target.alias]).issubset(set(columns)):
            message = f"Missing categorical columns {set(self.config.target.name).difference(set(columns))} after preprocessing"
            logger.info(message)
            raise ValueError(message)

        self._check_null_values()

        logger.info("Data quality check passed")

    def _check_null_values(self) -> None:
        """Check for null values in the DataFrame."""
        # check if there are any null values
        if self.df.isnull().any().any():
            message = "Unexpected null values found after preprocessing"
            logger.info(message)
            raise ValueError(message)

        logger.info("No null values found after preprocessing")

    def _apply_value_correction(self, correction: dict) -> None:
        """
        Apply value corrections to the DataFrame based on the provided dictionary.

        :param correction: A dictionary containing the corrections to be applied.
        """
        for column, values in correction.items():
            for old_value, new_value in values.items():
                self.df.loc[self.df[column] == old_value, column] = new_value

        logger.info("Value corrections applied successfully")

    def process_data(self) -> pd.DataFrame:
        """
        Process the data by renaming, validating columns, and checking data types.

        :return: A DataFrame containing the processed data.
        :rtype: pd.DataFrame
        """
        try:
            logger.info("Starting data preprocessing")
            self._validate_columns()
            self._rename_columns()
            self._convert_column_types()
            self._validate_column_types()

            corrections = {
                "Education": {0: 4, 5: 4, 6: 4},
                "Marriage": {0: 3},
                "Pay_0": {-1: 0, -2: 0},
                "Pay_2": {-1: 0, -2: 0},
                "Pay_3": {-1: 0, -2: 0},
                "Pay_4": {-1: 0, -2: 0},
                "Pay_5": {-1: 0, -2: 0},
                "Pay_6": {-1: 0, -2: 0},
                "Pay_amt1": {-1: 0, -2: 0},
                "Pay_amt2": {-1: 0, -2: 0},
                "Pay_amt3": {-1: 0, -2: 0},
                "Pay_amt4": {-1: 0, -2: 0},
                "Pay_amt5": {-1: 0, -2: 0},
                "Pay_amt6": {-1: 0, -2: 0},
            }

            self._apply_value_correction(correction=corrections)
            self._validate_processed_data()

            logger.info("Data cleaning completed successfully")
            logger.info(f"Final data shape: {self.df.shape}")
            logger.info(f"Final columns: {self.df.columns.tolist()}")
            logger.info(f"ID column data type: {self.df['Id'].dtype}")
            logger.info(f"Sample of cleaned data:\n{self.df.head().to_string()}")
            logger.info("Data cleaning script completed successfully")

            self.processed = True
            return self.df
        except ValidationError as err:
            logger.error(f"Configuration validation error: {err}")
            raise
        except Exception as err:
            logger.error(f"Error processing data: {err}")
            raise

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :random_state: The seed used by the random number generator.
        :return: A tuple containing the training set and the testing set.
        """
        if not self.processed:
            raise ValueError("Data must be processed before splitting.")

        X = self.df.drop(columns=[self.config.target.alias])  # pylint: disable=invalid-name
        y = self.df[self.config.target.alias]  # pylint: disable=invalid-name

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        train_set = pd.concat([X_train, y_train], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)

        logger.info(f"Data split into train and test sets with shapes {train_set.shape} and {test_set.shape}")

        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """
        Save the training and testing sets to a catalog.

        :param train_set: The training DataFrame to save.
        :param test_set: The testing DataFrame to save.
        :return: None
        """
        if not is_databricks():
            raise ValueError("This function is only supported on Databricks.")

        spark = SparkSession.builder.getOrCreate()

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", F.to_utc_timestamp(F.current_timestamp(), "UTC")
        )
        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", F.to_utc_timestamp(F.current_timestamp(), "UTC")
        )

        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.config.catalog_name}.{self.config.schema_name}")
        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )
        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        logger.info("Data saved to catalog successfully")


if __name__ == "__main__":
    print("Data cleaning using Spark...")
