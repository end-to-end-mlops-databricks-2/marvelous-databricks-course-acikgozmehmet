"""Unit tests for the DataCleaner class."""

import pathlib

import pandas as pd
import pytest
from pyspark.errors import AnalysisException

from credit_default.config import Config
from credit_default.data_ingestion import DataLoader
from credit_default.utility import is_databricks

# from tests.consts import PROJECT_DIR
from tests.consts import PROJECT_DIR


def test__load_data_success(dataloader) -> None:
    """
    Test the data loading functionality.

    This function verifies that the data is loaded correctly from the specified
    CSV file and that the DataFrame is not empty.

    :param None: This function does not take any parameters.
    """
    assert dataloader.df is not None
    assert dataloader.df.shape[0] > 0


@pytest.mark.skipif(is_databricks(), reason="Only Local test")
def test__load_data_file_not_found() -> None:
    """
    Test that a FileNotFoundError is raised when the data file is not found.

    :return: None
    :raises FileNotFoundError: If the data file does not exist.
    """
    data_filepath = PROJECT_DIR / "data" / "data2.csv"
    config_file_path = PROJECT_DIR / "project_config.yml"
    config = Config.from_yaml(config_file_path.as_posix())
    with pytest.raises(FileNotFoundError):
        _ = DataLoader(filepath=data_filepath.as_posix(), config=config)


@pytest.mark.skipif(not is_databricks(), reason="Only on Databricks")
def test__load_data_path_not_found() -> None:
    """
    Test that a FileNotFoundError is raised when the data file is not found.

    :return: None
    :raises FileNotFoundError: If the data file does not exist.
    """
    data_filepath = PROJECT_DIR / "data" / "data2.csv"
    config_file_path = PROJECT_DIR / "project_config.yml"
    config = Config.from_yaml(config_file_path.as_posix())
    with pytest.raises(AnalysisException):
        _ = DataLoader(filepath=data_filepath.as_posix(), config=config)


def test__validate_columns(dataloader) -> None:
    """
    Test the validation of columns in the data cleaner.

    This function loads the configuration and initializes the DataCleaner
    with the specified data file path, then validates the columns.

    :param None: This function does not take any parameters.
    :return: None
    """
    dataloader._validate_columns()


def test__validate_columns_missing_column_should_fail(dataloader) -> None:
    """
    Test to ensure it raises an exception when required columns are missing.

    This test checks that an exception is raised when the 'LIMIT_BAL' column is dropped
    from the DataFrame, indicating that the validation for required columns is functioning correctly.

    :raises Exception: If required columns are missing from the DataFrame.
    """
    dataloader.df.drop(columns=["LIMIT_BAL"], inplace=True)
    with pytest.raises(Exception) as exc:
        dataloader._validate_columns()
    assert "Missing columns in data:" in str(exc.value)


def test__validate_column_types_should_pass(dataloader) -> None:
    """
    Test the data types validation process to ensure it passes.

    This function sets up the necessary file paths and configuration,
    then invokes the data type validation method of the DataCleaner class.

    :param None: This function does not take any parameters.
    :return: None
    """
    dataloader._validate_columns()
    dataloader._rename_columns()
    dataloader._convert_column_types()
    dataloader._validate_column_types()


def test__validate_column_types_fail_if_dtype_not_matching(dataloader):
    """
    Test that the data type validation fails if the data types do not match.

    This function checks if an exception is raised when the numeric column's
    data type is not as expected.

    :raises Exception: If the data type of the numeric column does not match
                      the expected data type.
    """
    dataloader._validate_columns()
    dataloader._rename_columns()
    dataloader._convert_column_types()
    # to mess up the data types for test
    num_feature1 = dataloader.config.num_features[1].alias
    dataloader.df[num_feature1] = dataloader.df[num_feature1].astype("str")

    with pytest.raises(Exception) as exc:
        dataloader._validate_column_types()
    assert f"Numeric column '{num_feature1}' must be {dataloader.config.num_features[1].dtype}" in str(exc.value)


def test__validate_columnn_types_target_fail_if_dtype_not_matching(dataloader):
    """
    Test that the column type validation fails if the target column's data type does not match the expected type.

    This function checks if an exception is raised when the target column's data type is incorrect.

    :param None: This function does not take any parameters.
    :raises Exception: If the target column's data type does not match the expected type.
    """
    dataloader._rename_columns()
    dataloader._convert_column_types()
    # to mess up the target type for testing
    dataloader.df[dataloader.config.target.alias] = dataloader.df[dataloader.config.target.alias].astype("str")

    with pytest.raises(Exception) as exc:
        dataloader._validate_column_types()
    assert f"Target column '{dataloader.config.target.alias}' must be {dataloader.config.target.dtype}" in str(
        exc.value
    )


@pytest.mark.skipif(is_databricks(), reason="Only Local test")
def test__apply_value_correction(tmp_path: pathlib.Path) -> None:
    """
    Test the application of value corrections to a DataFrame.

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


def test__convert_column_types(dataloader) -> None:
    """
    Test the assignment of column types to DataFrame columns.

    This function loads the configuration and data, processes the data,
    and assigns the data types to the DataFrame columns.
    """
    dataloader._rename_columns()

    # LIMIT_BAL -> 1st numeric feature
    dataloader.config.num_features[1].dtype = "int64"

    dataloader._convert_column_types()

    assert dataloader.df["Limit_bal"].dtype == "int64"


def test_process_data(dataloader) -> None:
    """
    Test the process of data cleaning.

    This function loads configuration and processes the data using the DataCleaner class.

    :param None: This function does not take any parameters.
    """
    dataloader.process_data()
    assert dataloader.df is not None
    assert dataloader.processed is True


def test_target_value_counts(dataloader) -> None:
    """
    Test the value counts of the target variable in the dataset.

    This function loads the configuration and data, processes the data,
    and prints the normalized value counts of the target variable.

    :param None: This function does not take any parameters.
    :return: None
    """
    dataloader.process_data()
    target = dataloader.config.target.alias
    target_counts = dataloader.df[target].value_counts(normalize=True)
    print(f"{target_counts =}")


def test_split_data_when_processed_data(dataloader) -> None:
    """Test the split_data method of the DataPreprocessor class."""
    dataloader.process_data()
    train_set, test_set = dataloader.split_data()
    assert train_set is not None
    assert train_set.shape[0] > 0
    assert train_set.shape[1] == len(dataloader.df.columns)
    assert test_set is not None
    assert test_set.shape[1] == len(dataloader.df.columns)


def test_split_data_when_not_processed_data(dataloader) -> None:
    """
    Test the split_data method when the data has not been processed.

    :param data_cleaner: An instance of the data cleaner class.
    :raises ValueError: If the data has not been processed.
    """
    with pytest.raises(ValueError) as exc:
        _, _ = dataloader.split_data()
    assert "Data must be processed before splitting." in str(exc.value)
