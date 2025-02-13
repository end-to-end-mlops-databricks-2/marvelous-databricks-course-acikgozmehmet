"""Data cleaning for credit default data using Spark."""
# pylint disable=invalid-name

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import ValidationError
from pyspark.errors import AnalysisException
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split

from hotel_reservations.config import Config
from hotel_reservations.utility import is_databricks, normalize_arrival_date


# pylint disable=invalid-name
class DataLoader:
    """A class for cleaning and processing data.

    :param filepath: The path to the data file to be loaded.
    :param config: Configuration object for data cleaning.
    """

    def __init__(self, filepath: str, config: Config) -> None:
        """Initialize the DataCleaner with the given file path and configuration.

        :param filepath: The path to the data file to be loaded.
        :param config: Configuration object for data cleaning.
        """
        self.df = self._load_data(filepath)  # pylint: disable=invalid-name
        self.config = config
        self.config.cat_features = [] if self.config.cat_features is None else self.config.cat_features
        self.processed: bool = False

        logger.info(f"DataLoader initialized with {filepath} and configuration")

    def _load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from a CSV file into a DataFrame.

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

    def _validate_required_columns(self) -> None:
        """Validate the presence of required columns in the DataFrame.

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
            raise ValueError(  # pylint: disable=broad-exception-raised
                f"Missing columns in data: {missing_columns})"
            )
        logger.info("All required columns are present in the data")

    def _rename_columns(self) -> None:
        """Rename the columns in the DataFrame based on the configuration.

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

    def _convert_column_data_types(self) -> None:
        """Convert data types of the DataFrame columns based on the configuration.

        Pandas will automatically infer the data types for each column while reading a CSV file.
        If dtypes are not specified dtypes, pandas will:
            Infer numeric types (int64, float64) for columns with numbers
            Use object type for columns with mixed data or strings
            Attempt to parse dates if the data looks like dates
        This method updates the DataFrame's columns to the specified data types
        for numerical, categorical features, and the target variable after reading the CSV file .
        """
        features = self.config.num_features + self.config.cat_features
        for ele in features + [self.config.target]:
            self.df[ele.alias] = self.df[ele.alias].astype(ele.dtype)
            if ele.dtype == "category":
                ele.dtype = pd.CategoricalDtype()

        logger.info("Columns converted to specified data types")

    def _validate_data_types(self) -> None:
        """Validate that the data types of the columns match the specified types in the configuration."""
        features = self.config.num_features + self.config.cat_features
        for ele in features + [self.config.target]:
            column_dtype = self.df[ele.alias].dtype
            expected_dtype = ele.dtype

            if expected_dtype == "category" or isinstance(expected_dtype, pd.CategoricalDtype):
                if not isinstance(column_dtype, pd.CategoricalDtype):
                    raise ValueError(f"Column {ele.alias} should be of type 'category', but is {column_dtype}")
            else:
                if not np.issubdtype(column_dtype, np.dtype(expected_dtype)):
                    raise ValueError(f"Column {ele.alias} should be of type {expected_dtype}, but is {column_dtype}")

        logger.info("Data types validation passed")

    def _validate_processed_data(self) -> None:
        """Validate the processed data to ensure it meets the required quality checks.

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

        if not {self.config.target.alias}.issubset(set(columns)):
            message = (
                f"Missing target column {set(self.config.target.name).difference(set(columns))} after preprocessing"
            )
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
        """Apply value corrections to the DataFrame based on the provided dictionary.

        :param correction: A dictionary containing the corrections to be applied.
        """
        for column, values in correction.items():
            for old_value, new_value in values.items():
                self.df.loc[self.df[column] == old_value, column] = new_value

        logger.info("Value corrections applied successfully")

    def _normalize_arrival_date(self) -> None:
        """Normalize the arrival date in the dataframe.

        Applies the normalize_arrival_date function to each row of the dataframe and updates the 'date_of_arrival' column.
        """
        self.df["date_of_arrival"] = self.df.apply(normalize_arrival_date, axis=1)
        logger.info("Arrival date normalized to 'date_of_arrival' successfully.")

    def _create_date_of_booking_column(self) -> None:
        """Create a new column 'date_of_booking' by subtracting 'lead_time' from 'date_of_arrival'."""
        self.df["date_of_arrival"] = pd.to_datetime(self.df["date_of_arrival"])
        self.df["lead_time"] = pd.to_timedelta(self.df["lead_time"], unit="D")
        self.df["date_of_booking"] = self.df["date_of_arrival"] - self.df["lead_time"]

        logger.info("The column 'date_of_booking' created successfully.")

    def process_data(self) -> pd.DataFrame:
        """Process the data by renaming, validating columns, and checking data types.

        :return: A DataFrame containing the processed data.
        :rtype: pd.DataFrame
        """
        try:
            logger.info("Starting data preprocessing")
            # Default operations on data preprocessing
            self._validate_required_columns()
            self._rename_columns()
            corrections = {"booking_status": {"Canceled": 1, "Not_Canceled": 0}}
            self._apply_value_correction(corrections)

            self._convert_column_data_types()
            self._validate_data_types()

            # Custom operations on data preprocessing
            self._normalize_arrival_date()
            self._create_date_of_booking_column()

            # final checks on data preprocessing
            self._validate_processed_data()

            logger.info("Data cleaning completed successfully")
            logger.info(f"Final data shape: {self.df.shape}")
            logger.info(f"Final columns: {self.df.columns.tolist()}")
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

    def split_and_extract_data(self) -> pd.DataFrame:
        """Split the dataframe and extract non-online market segment data.

        Filters out rows where market_segment_type is not "Online" and returns them as a new DataFrame.

        :param self: The instance of the class containing this method.
        :return: A DataFrame containing rows where market_segment_type is not "Online".
        """
        extra = self.df.query('market_segment_type != "Online"')
        self.df = self.df.query('market_segment_type == "Online"')
        return extra

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into training and testing sets.

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

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, extra_set: pd.DataFrame = None) -> None:
        """Save the training and testing sets to a catalog.

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

        # spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.config.catalog_name}.{self.config.schema_name}") #noqa ERA

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

        if extra_set is not None:
            extra_set_with_timestamp = spark.createDataFrame(extra_set).withColumn(
                "update_timestamp_utc", F.to_utc_timestamp(F.current_timestamp(), "UTC")
            )
            extra_set_with_timestamp.write.mode("overwrite").saveAsTable(
                f"{self.config.catalog_name}.{self.config.schema_name}.extra_set"
            )
            spark.sql(
                f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.extra_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
            )
            logger.info("Extra data saved to catalog successfully as well.")

        logger.info("Train and test data saved to catalog successfully")


if __name__ == "__main__":
    print("Data cleaning using Spark...")
