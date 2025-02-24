"""Unit tests for DataFabricator."""

import pathlib

import pandas as pd
import pytest

from hotel_reservations.data_ingestion import DataLoader
from hotel_reservations.synthetic import DataFabricator


def test_init_happy_path(dataloader: DataLoader) -> None:
    """Test the initialization of DataFabricator with a valid DataLoader.

    :param dataloader: An instance of DataLoader
    """
    fabricator = DataFabricator(dataloader)
    assert fabricator.payload is not None
    assert fabricator.payload is not dataloader  # Should be a deep copy


def test_setattr_immutability(dataloader: DataLoader) -> None:
    """Test that new attributes cannot be added to DataFabricator.

    :param dataloader: An instance of DataLoader
    """
    fabricator = DataFabricator(payload=dataloader)
    with pytest.raises(AttributeError):
        fabricator.new_attribute = "value"


def test_delattr_prevention(dataloader: DataLoader) -> None:
    """Test that attributes cannot be deleted from DataFabricator.

    :param dataloader: An instance of DataLoader
    """
    fabricator = DataFabricator(payload=dataloader)
    with pytest.raises(AttributeError):
        del fabricator._payload


def test_synthesize_happy_path(dataloader: DataLoader) -> None:
    """Test the synthesize method with valid input.

    :param dataloader: An instance of DataLoader
    """
    fabricator = DataFabricator(payload=dataloader)
    expected_columns = dataloader.df.columns.tolist()

    result = fabricator.synthesize(num_rows=5)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5
    assert "Booking_ID" in result.columns
    assert set(result.columns.tolist()) == set(expected_columns)


def test_synthesize_empty_dataframe(dataloader: DataLoader) -> None:
    """Test synthesize method with an empty DataFrame.

    :param dataloader: An instance of DataLoader
    """
    dataloader.df = pd.DataFrame()
    fabricator = DataFabricator(dataloader)
    with pytest.raises(Exception):  # noqa  # Expect some kind of error when df is empty
        fabricator.synthesize()


def test_synthesize_missing_columns(dataloader: DataLoader) -> None:
    """Test synthesize method with missing required columns.

    :param dataloader: An instance of DataLoader
    """
    del dataloader.df["no_of_adults"]
    fabricator = DataFabricator(dataloader)
    with pytest.raises(Exception):  # noqa # Expect error when required column is missing
        fabricator.synthesize()


def test_to_csv(dataloader: DataLoader, tmp_path: pathlib.Path) -> None:
    """Test the to_csv method of DataFabricator.

    :param dataloader: An instance of DataLoader
    :param tmp_path: A temporary path for file creation
    """
    fabricator = DataFabricator(dataloader)
    output_file = tmp_path / "test_output.csv"
    df = fabricator.payload.df.copy()
    fabricator.to_csv(df, str(output_file))
    assert output_file.exists()
    assert pd.read_csv(output_file).equals(df)


def test_synthesize_respects_column_order(dataloader: DataLoader) -> None:
    """Test if synthesize method respects the original column order.

    :param dataloader: An instance of DataLoader
    """
    original_order = dataloader.df.columns.tolist()
    fabricator = DataFabricator(dataloader)
    result = fabricator.synthesize(num_rows=5)
    assert result.columns.tolist() == original_order


def test_synthesize(dataloader: DataLoader, tmp_path: pathlib.Path) -> None:
    """Test the synthesize method of the dataloader.

    This function tests if the synthesize method of the dataloader generates
    the correct number of rows and columns in the output DataFrame.

    :param dataloader: The dataloader object with a synthesize method and a df attribute
    :param tmp_path: A temporary path object for creating the output file
    """
    fabricator = DataFabricator(payload=dataloader)
    expected_col_number = fabricator.payload.df.shape[1]
    num_rows = 10
    # output_filename = (tmp_path / "delete_me.csv").as_posix() # noqa
    actual_df = fabricator.synthesize(num_rows=num_rows)

    # actual_df = pd.read_csv(output_filename) # noqa
    actual_shape = actual_df.shape
    assert actual_shape == (10, expected_col_number)
