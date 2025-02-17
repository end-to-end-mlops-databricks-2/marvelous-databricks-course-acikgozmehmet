"""Serving Module."""

import datetime
import time
from loguru import logger

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceConflict
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup

from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

# model_name -> full_model_name


class ServingBase:
    """A base class for managing serving endpoints in Databricks.

    This class provides methods to deploy or update serving endpoints using the Databricks SDK.
    """

    def __init__(self, endpoint_name: str) -> None:
        """Initialize the ServingBase instance.

        :param endpoint_name: The name of the serving endpoint
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name

    def deploy_or_update_serving_endpoint(self, served_entities: list[ServedEntityInput]) -> None:
        """Deploy a new serving endpoint or update an existing one.

        This method checks if the endpoint exists and creates a new one if it doesn't,
        or updates the existing endpoint if it already exists.

        :param served_entities: A list of ServedEntityInput objects representing the entities to be served
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)

    def delete_serving_endpoint(self):
        """Delete the serving endpoint."""
        try:
            self.workspace.serving_endpoints.delete(name=self.endpoint_name)
            logger.info(f"Serving endpoint '{self.endpoint_name}' deleted successfully")
        except ResourceConflict:
            logger.error(f"Failed to delete serving endpoint '{self.endpoint_name}': Resource conflict")


class ModelServing(ServingBase):
    """A class for managing model serving operations.

    This class extends ServingBase to handle deployment and updates of model serving endpoints.
    """

    def __init__(
        self, model_name: str, endpoint_name: str, workload_size: str = "Small", scale_to_zero: bool = True
    ) -> None:
        """Initialize the ModelServing instance.

        :param model_name: Name of the model to be served
        :param endpoint_name: Name of the serving endpoint
        :param workload_size: Size of the workload, defaults to "Small"
        :param scale_to_zero: Whether to enable scale-to-zero, defaults to True
        """
        super().__init__(endpoint_name)
        # full_model_name = model_name change this
        self.model_name = model_name
        entity_version = self._get_latest_model_version()
        self.served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

    def _get_latest_model_version(self) -> str:
        """Retrieve the latest version of the model.

        :return: The latest version of the model as a string
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        logger.info(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint_with_retry(self, max_retries=10, retry_interval=30):
        super().deploy_or_update_serving_endpoint(self.served_entities)
        for attempt in range(max_retries):
            try:
                self.workspace.serving_endpoints.update_config_and_wait(
                    name=f"{self.endpoint_name}", served_entities=self.served_entities,
                    timeout=datetime.timedelta(minutes=20)
                )
                logger.info("Deployment successful")
                return
            except ResourceConflict as e:
                if "Endpoint served entities are currently being updated" in str(e):
                    logger.info(f"Attempt {attempt + 1}/{max_retries}: Endpoint is being updated. Retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)
                else:
                    raise
        logger.info("Max retries reached. Deployment failed.")


class FeatureServing(ServingBase):
    def __init__(self, endpoint_name: str,
                 feature_table_name: str,
                 feature_spec_name: str,
                 workload_size: str = "Small",
                 scale_to_zero: bool = True):
        super().__init__(endpoint_name)
        self.feature_table_name = feature_table_name
        self.feature_spec_name = feature_spec_name
        self.online_table_name = f"{feature_table_name}_online"
        self.fe = feature_engineering.FeatureEngineeringClient()
        self.served_entities = [
            ServedEntityInput(
                entity_name=self.feature_spec_name, scale_to_zero_enabled=scale_to_zero, workload_size=workload_size
            )
        ]

    def create_online_table(self):
        spec = OnlineTableSpec(
            primary_key_columns=["booking_id"],
            source_table_full_name=self.feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )
        self.workspace.online_tables.create(name=self.online_table_name, spec=spec)

    def _create_feature_spec(self):
        # "lead_time" in feature_names??
        features = [
            FeatureLookup(
                table_name=self.feature_table_name,
                lookup_key="booking_id",
                feature_names=["repeated_guest", "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"],
            )
        ]
        self.fe.create_feature_spec(name=self.feature_spec_name, features=features, exclude_columns=None)

    def deploy_or_update_serving_endpoint_with_retry(self, max_retries=10, retry_interval=30):
        super().deploy_or_update_serving_endpoint(self.served_entities)
        for attempt in range(max_retries):
            try:
                self.workspace.serving_endpoints.update_config_and_wait(
                    name=f"{self.endpoint_name}", served_entities=self.served_entities,
                    timeout=datetime.timedelta(minutes=20)
                )
                logger.info("Deployment successful")
                return
            except ResourceConflict as e:
                if "Endpoint served entities are currently being updated" in str(e):
                    logger.info(f"Attempt {attempt + 1}/{max_retries}: Endpoint is being updated. Retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)
                else:
                    raise
        logger.info("Max retries reached. Deployment failed.")
