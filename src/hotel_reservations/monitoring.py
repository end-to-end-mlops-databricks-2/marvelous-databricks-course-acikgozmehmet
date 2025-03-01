"""Monitoring module."""

from databricks.sdk.service.serving import QueryEndpointResponse

from hotel_reservations.feature_lookup_model import FeatureLookUpModel
from hotel_reservations.serving import FeatureLookupServing


class Monitor:
    """A class for monitoring and interacting with model serving endpoints."""

    def __init__(self, model: FeatureLookUpModel, serving: FeatureLookupServing) -> None:
        """Initialize the Monitor with a model and serving object.

        :param model: The model to be monitored
        :param serving: The serving object to interact with endpoints
        """
        self.model = model
        self.serving = serving

    def query_request(self, dataframe_record: dict) -> QueryEndpointResponse:
        """Send a query request to the serving endpoint.

        :param dataframe_record: A dictionary representing the dataframe record to be queried
        :return: The response from the serving endpoint
        """
        response = self.serving.workspace.serving_endpoints.query(
            name=self.serving.endpoint_name, dataframe_records=[dataframe_record]
        )
        return response

    def create_or_refresh(self) -> None:
        """Create or refresh the monitor table.

        This method is currently a placeholder and does not perform any operations.
        """
        pass
