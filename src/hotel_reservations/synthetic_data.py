"""Synthetic data genetarion module."""

import copy
import uuid

import numpy as np
import pandas as pd
from loguru import logger

from hotel_reservations.data_ingestion import DataLoader


class DataFabricator:
    """A class for generating synthetic data based on an original dataset.

    This class takes a DataLoader object as input and provides methods to synthesize
    new data while maintaining the characteristics of the original dataset.
    """

    def __init__(self, payload: DataLoader) -> None:
        """Initialize the DataFabricator with a DataLoader object.

        :param payload: A DataLoader object containing the original dataset
        :raises ValueError: If the DataLoader has already been processed
        """
        if payload.processed:
            message = "Dataloader should not have been processed before creating DataFabricator."
            logger.error(message)
            raise ValueError(message)

        self._payload = copy.deepcopy(payload)

    @property
    def payload(self) -> DataLoader:
        """Get a deep copy of the payload.

        :return: A deep copy of the DataLoader object
        """
        return copy.deepcopy(self._payload)

    def __setattr__(self, name: str, value: object) -> None:
        """Control attribute assignment to ensure immutability of certain attributes.

        :param name: Name of the attribute
        :param value: Value to be assigned
        :raises AttributeError: If attempting to modify an immutable attribute
        """
        if name == "_payload":
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot modify attribute '{name}' of immutable DataFrame.")

    def __delattr__(self, name: str) -> None:
        """Prevent deletion of attributes.

        :param name: Name of the attribute
        :raises AttributeError: Always, as attribute deletion is not allowed
        """
        raise AttributeError(f"Cannot delete attribute '{name}' of immutable DataFrame")

    def synthesize(self, num_rows: int = 10) -> pd.DataFrame:
        """Generate synthetic data based on the original dataframe.

        This method creates synthetic features, validates columns, and generates synthetic data.

        :param num_rows: The number of rows to generate in the synthetic dataset
        :return: A pandas DataFrame containing the generated synthetic data
        """
        payload = self.payload

        # columns order
        columns_order = payload.df.columns.tolist()

        #  drop nan values from df
        payload.df.dropna(inplace=True)

        #  _validate_required_columns
        payload._validate_required_columns()

        # rename columns
        payload._rename_columns()

        # convert data types
        payload._convert_column_data_types()

        # query only market_segment_type = 'online'
        payload.df = payload.df.query("market_segment_type == 'Online'")

        # generate synthetic data
        payload.df = self._generate_synthetic_data(payload, num_rows)

        # convert data types
        payload._convert_column_data_types()

        # validate_data_types
        payload._validate_data_types()

        # back to original column names
        for col in payload.config.num_features + payload.config.cat_features + [payload.config.target]:
            payload.df.rename(columns={col.alias: col.name}, inplace=True)

        # reorder  the columns
        payload.df = payload.df[columns_order]

        return payload.df

    @staticmethod
    def to_csv(dataframe: pd.DataFrame, output_filename: str) -> None:
        """Save a pandas DataFrame to a CSV file.

        :param dataframe: The DataFrame to be saved
        :param output_filename: The name of the output CSV file
        """
        dataframe.to_csv(output_filename, index=False)
        logger.info(f"Dataframe saved to {output_filename}")

    def _generate_synthetic_data(self, payload: DataLoader, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data based on the configuration and original dataframe.

        This method creates synthetic data for numerical and categorical features,
        including a special case for generating unique booking IDs.

        :param payload: The DataLoader object containing the original data and configuration
        :param num_rows: The number of rows to generate in the synthetic dataset
        :return: A pandas DataFrame containing the generated synthetic data
        """
        # synthetic data logic implementation
        synthetic_data = pd.DataFrame()
        for col in payload.config.num_features:
            synthetic_data[col.alias] = np.round(
                np.abs(
                    np.random.normal(self.payload.df[col.alias].mean(), self.payload.df[col.alias].std(), size=num_rows)
                ),
                decimals=2,
            )
        rng = np.random.default_rng()
        for col in payload.config.cat_features + [payload.config.target]:
            if col.alias == "booking_id":
                synthetic_data["booking_id"] = [str(uuid.uuid4()) for i in range(num_rows)]
            else:
                # Get unique values and their probabilities
                unique_values = payload.df[col.alias].unique()
                probabilities = payload.df[col.alias].value_counts(normalize=True).reindex(unique_values).fillna(0)

                # Ensure probabilities sum to 1
                probabilities = probabilities / probabilities.sum()

                synthetic_data[col.alias] = rng.choice(unique_values, size=num_rows, p=probabilities)
        return synthetic_data
