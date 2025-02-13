"""Unit tests for the DataCleaner class."""

import pathlib
from datetime import date

import pandas as pd
import pytest
from pyspark.errors import AnalysisException
from pytest import CaptureFixture

from hotel_reservations.config import Config
from hotel_reservations.data_ingestion import DataLoader
from hotel_reservations.utility import is_databricks
from tests.consts import PROJECT_DIR


def test__load_data_success(dataloader: DataLoader) -> None:
    """Test the data loading functionality.

    This function verifies that the data is loaded correctly from the specified
    CSV file and that the DataFrame is not empty.

    :param None: This function does not take any parameters.
    """
    assert dataloader.df is not None
    assert dataloader.df.shape[0] > 0


@pytest.mark.skipif(is_databricks(), reason="Only Local test")
def test__load_data_file_not_found() -> None:
    """Test that a FileNotFoundError is raised when the data file is not found.

    :return: None
    :raises FileNotFoundError: If the data file does not exist.
    """
    data_filepath = PROJECT_DIR / "data" / "data2.csv"
    config_file_path = PROJECT_DIR / "project_config.yml"
    config = Config.from_yaml(config_file_path.as_posix())
    with pytest.raises(FileNotFoundError):
        _ = DataLoader(filepath=data_filepath.as_posix(), config=config)


@pytest.mark.skipif(not is_databricks(), reason="Only runs on Databricks")
def test__load_data_path_not_found() -> None:
    """Test that a FileNotFoundError is raised when the data file is not found.

    :return: None
    :raises FileNotFoundError: If the data file does not exist.
    """
    data_filepath = PROJECT_DIR / "data" / "data2.csv"
    config_file_path = PROJECT_DIR / "project_config.yml"
    config = Config.from_yaml(config_file_path.as_posix())
    with pytest.raises(AnalysisException):
        _ = DataLoader(filepath=data_filepath.as_posix(), config=config)


def test___validate_required_columns(dataloader: DataLoader, capsys: CaptureFixture) -> None:
    """Test the validation of columns in the data cleaner.

    This function loads the configuration and initializes the DataCleaner
    with the specified data file path, then validates the columns.

    :param None: This function does not take any parameters.
    :return: None
    """
    dataloader._validate_required_columns()
    captured = capsys.readouterr()
    assert captured.out == ""


def test__validate_columns_missing_column_should_fail(dataloader: DataLoader) -> None:
    """Test to ensure it raises an exception when required columns are missing.

    This test checks that an exception is raised when the 'Booking_ID' column is dropped
    from the DataFrame, indicating that the validation for required columns is functioning correctly.

    :raises ValueError: If required columns are missing from the DataFrame.
    """
    dataloader.df.drop(columns=["Booking_ID"], inplace=True)
    with pytest.raises(ValueError) as exc:
        dataloader._validate_required_columns()
    assert "Missing columns in data:" in str(exc.value)


def test__validate_data_types_should_pass(dataloader: DataLoader) -> None:
    """Test the data types validation process to ensure it passes.

    This function sets up the necessary file paths and configuration,
    then invokes the data type validation method of the DataCleaner class.

    :param None: This function does not take any parameters.
    :return: None
    """
    dataloader._validate_required_columns()
    dataloader._rename_columns()
    dataloader._convert_column_data_types()
    dataloader._validate_data_types()
    print()


def test__validate_column_types_fail_if_dtype_not_matching(dataloader: DataLoader) -> None:
    """Test that the data type validation fails if the data types do not match.

    This function checks if an exception is raised when the numeric column's
    data type is not as expected.

    :raises Exception: If the data type of the numeric column does not match
                      the expected data type.
    """
    dataloader._validate_required_columns()
    dataloader._rename_columns()
    dataloader._convert_column_data_types()
    # to mess up the data types for test
    num_feature1 = dataloader.config.num_features[1].alias
    dataloader.df[num_feature1] = dataloader.df[num_feature1].astype("str")

    with pytest.raises(Exception) as exc:
        dataloader._validate_data_types()
    assert (
        f"Column no_of_children should be of type {dataloader.config.num_features[1].dtype}, but is {dataloader.df[num_feature1].dtype}"
        in str(exc.value)
    )


def test__validate_columnn_types_target_fail_if_dtype_not_matching(dataloader: DataLoader) -> None:
    """Test that the column type validation fails if the target column's data type does not match the expected type.

    This function checks if an exception is raised when the target column's data type is incorrect.

    :param None: This function does not take any parameters.
    :raises Exception: If the target column's data type does not match the expected type.
    """
    dataloader._rename_columns()
    dataloader._convert_column_data_types()
    # to mess up the target type for testing
    dataloader.df[dataloader.config.target.alias] = dataloader.df[dataloader.config.target.alias].astype("str")

    with pytest.raises(Exception) as exc:
        dataloader._validate_data_types()
    assert (
        f"Column {dataloader.config.target.alias} should be of type '{dataloader.config.target.dtype}', but is {dataloader.df[dataloader.config.target.alias].dtype}"
        in str(exc.value)
    )


@pytest.mark.skipif(is_databricks(), reason="Only Local test")
def test__apply_value_correction(tmp_path: pathlib.Path) -> None:
    """Test the application of value corrections to a DataFrame.

    :param tmp_path: A temporary directory path for creating test files.
    :type tmp_path: pathlib.Path
    """
    config_file_path = PROJECT_DIR / "project_config.yml"
    config = Config.from_yaml(config_file_path.as_posix())

    temp_file = tmp_path / "temp.csv"
    data = [[0, 0, -1], [5, 1, -2], [6, 0, 0]]
    columns = ["Education", "Marriage", "Pay"]
    dataframe = pd.DataFrame(data=data, columns=columns)
    dataframe.to_csv(temp_file.as_posix(), index=False)

    dataloader = DataLoader(filepath=temp_file.as_posix(), config=config)
    corrections = {
        "Education": {0: 4, 5: 4, 6: 4},
        "Marriage": {0: 3},
        "Pay": {-1: 0, -2: 0},
    }

    expected = {
        "Education": {0: 4, 1: 4, 2: 4},
        "Marriage": {0: 3, 1: 1, 2: 3},
        "Pay": {0: 0, 1: 0, 2: 0},
    }

    dataloader._apply_value_correction(correction=corrections)
    actual = dataloader.df.to_dict()
    assert actual == expected


def test__convert_column_types(dataloader: DataLoader) -> None:
    """Test the assignment of column types to DataFrame columns.

    This function loads the configuration and data, processes the data,
    and assigns the data types to the DataFrame columns.
    """
    dataloader._rename_columns()

    # no_of_children -> 1st numeric feature
    dataloader.config.num_features[1].dtype = "int64"

    dataloader._convert_column_data_types()

    assert dataloader.df["no_of_children"].dtype == "int64"


def test_process_data(dataloader: DataLoader) -> None:
    """Test the process of data cleaning.

    This function loads configuration and processes the data using the DataCleaner class.

    :param None: This function does not take any parameters.
    """
    dataloader.process_data()
    assert dataloader.df is not None
    assert dataloader.processed is True


def test_target_value_counts(dataloader: DataLoader) -> None:
    """Test the value counts of the target variable in the dataset.

    This function loads the configuration and data, processes the data,
    and prints the normalized value counts of the target variable.

    :param None: This function does not take any parameters.
    :return: None
    """
    dataloader.process_data()
    target = dataloader.config.target.alias
    target_counts = dataloader.df[target].value_counts(normalize=True)
    print(f"{target_counts =}")


def test_split_data_when_processed_data(dataloader: DataLoader) -> None:
    """Test the split_data method of the DataPreprocessor class."""
    dataloader.process_data()
    train_set, test_set = dataloader.split_data()
    assert train_set is not None
    assert train_set.shape[0] > 0
    assert train_set.shape[1] == len(dataloader.df.columns)
    assert test_set is not None
    assert test_set.shape[1] == len(dataloader.df.columns)


def test_split_data_when_not_processed_data(dataloader: DataLoader) -> None:
    """Test the split_data method when the data has not been processed.

    :param data_cleaner: An instance of the data cleaner class.
    :raises ValueError: If the data has not been processed.
    """
    with pytest.raises(ValueError) as exc:
        _, _ = dataloader.split_data()
    assert "Data must be processed before splitting." in str(exc.value)


def test__normalize_arrival_date(dataloader: DataLoader) -> None:
    """Test the _normalize_arrival_date method of the DataLoader class.

    This function checks if the method correctly adds an 'date_of_arrival' column to
    the dataframe and ensures all values in the column are of type date.

    :param dataloader: An instance of the DataLoader class to be tested
    """
    dataloader._normalize_arrival_date()
    assert "date_of_arrival" in dataloader.df.columns.tolist()
    assert all(isinstance(d, date) for d in dataloader.df["date_of_arrival"])


def test__create_date_of_booking_column(dataloader: DataLoader) -> None:
    """Test the creation of the date_of_booking column in the DataLoader.

    This function verifies that the date_of_booking column is correctly created and contains valid date objects.

    :param dataloader: The DataLoader instance to be tested
    """
    dataloader._normalize_arrival_date()
    assert "date_of_arrival" in dataloader.df.columns.tolist()

    dataloader._create_date_of_booking_column()
    assert "date_of_booking" in dataloader.df.columns.tolist()
    assert all(isinstance(d, date) for d in dataloader.df["date_of_booking"])

    expected_date_of_arrival = dataloader.df["date_of_booking"] + pd.to_timedelta(dataloader.df["lead_time"], unit="d")
    assert (dataloader.df["date_of_arrival"] == expected_date_of_arrival).all(), "The columns are not identical"


def test_split_and_extract_data(dataloader: DataLoader) -> None:
    """Test the split_and_extract_data method of the DataLoader class.

    This function checks if the method correctly splits the data into two sets based on a condition.

    :param dataloader: An instance of the DataLoader class to be tested
    """
    dataloader.process_data()
    extra_set = dataloader.split_and_extract_data()
    assert "Online" not in extra_set["market_segment_type"].unique().tolist()
    assert dataloader.df["market_segment_type"].nunique() == 1
    assert "Online" in dataloader.df["market_segment_type"].unique().tolist()
