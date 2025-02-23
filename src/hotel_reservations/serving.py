"""Serving Module."""

import time
from datetime import timedelta

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceConflict
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from loguru import logger

from hotel_reservations.config import Config

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

    def deploy_or_update_serving_endpoint_with_retry(
        self, served_entities: list[ServedEntityInput], max_retries: int = 10, retry_interval: int = 30
    ) -> None:
        """Deploy or update a serving endpoint with retry mechanism.

        This function attempts to deploy or update a serving endpoint, retrying in case of conflicts.

        :param served_entities: List of served entities to be deployed or updated
        :param max_retries: Maximum number of retry attempts
        :param retry_interval: Time interval between retries in seconds
        """
        self.deploy_or_update_serving_endpoint(served_entities)
        for attempt in range(max_retries):
            try:
                self.workspace.serving_endpoints.update_config_and_wait(
                    name=f"{self.endpoint_name}",
                    served_entities=served_entities,
                    timeout=timedelta(seconds=retry_interval),
                )
                logger.info("Deployment successful")
                return
            except ResourceConflict as e:
                if "Endpoint served entities are currently being updated" in str(e):
                    logger.info(
                        f"Attempt {attempt + 1}/{max_retries}: Endpoint is being updated. Retrying in {retry_interval} seconds..."
                    )
                    time.sleep(retry_interval)
                else:
                    raise
        logger.info("Max retries reached. Deployment failed.")

    def delete_serving_endpoint(self) -> None:
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
        self,
        endpoint_name: str,
        model_name: str,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
    ) -> None:
        """Initialize the ModelServing instance.

        :param endpoint_name: Name of the serving endpoint
        :param model_name: Name of the model to be served
        :param version: str. Version of the model to deploy
        :param workload_size: Size of the workload, defaults to "Small"
        :param scale_to_zero: Whether to enable scale-to-zero, defaults to True
        """
        super().__init__(endpoint_name)
        # full_model_name = model_name change this
        self.model_name = model_name
        entity_version = self.get_latest_model_version() if version == "latest" else version
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

    def deploy_or_update_serving_endpoint_with_retry(self, max_retries: int = 10, retry_interval: int = 30) -> None:
        """Deploy or update a serving endpoint with retry mechanism.

        :param max_retries: Maximum number of retry attempts
        :param retry_interval: Time interval between retries in seconds
        """
        super().deploy_or_update_serving_endpoint_with_retry(
            served_entities=self.served_entities, max_retries=max_retries, retry_interval=retry_interval
        )


class FeatureServing(ServingBase):
    """A class for managing feature serving in Databricks.

    This class handles the creation of online tables,
    feature specifications, and deployment of serving endpoints.
    """

    def __init__(
        self,
        endpoint_name: str,
        feature_table_name: str,
        feature_spec_name: str,
        workload_size: str = "Small",
        scale_to_zero: bool = True,
    ) -> None:
        super().__init__(endpoint_name)
        self.feature_table_name = feature_table_name
        self.feature_spec_name = feature_spec_name
        self.fe = feature_engineering.FeatureEngineeringClient()
        self.served_entities = [
            ServedEntityInput(
                entity_name=self.feature_spec_name, scale_to_zero_enabled=scale_to_zero, workload_size=workload_size
            )
        ]

    @staticmethod
    def form_online_table(feature_table_name: str, primary_keys: list[str]) -> None:
        """Create an online table from a feature table with a standard name.

        :param feature_table_name: The name of the source feature table
        :param primary_keys: A list of primary key column names
        """
        spec = OnlineTableSpec(
            primary_key_columns=primary_keys,
            source_table_full_name=feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )

        workspace = WorkspaceClient()
        online_table_name = f"{feature_table_name}_online"
        workspace.online_tables.create(name=online_table_name, spec=spec)

    def create_online_table(self) -> None:
        """Create an online table in Databricks.

        This method creates an online table using the specified feature table as the source.
        The online table is configured with a triggered scheduling policy and without performing a full copy.

        :param self: The instance of the class containing this method.
        """
        FeatureServing.form_online_table(feature_table_name=self.feature_table_name, primary_keys=["booking_id"])

    def create_feature_spec(self) -> None:
        """Create a feature specification for the model.

        This method defines the features to be used in the model by creating a FeatureLookup
        object and using it to create a feature specification.
        """
        # "lead_time" in feature_names??
        features = [
            FeatureLookup(
                table_name=self.feature_table_name,
                lookup_key="booking_id",
                feature_names=[
                    "repeated_guest",
                    "no_of_previous_cancellations",
                    "no_of_previous_bookings_not_canceled",
                ],
            )
        ]
        self.fe.create_feature_spec(name=self.feature_spec_name, features=features, exclude_columns=None)

    def deploy_or_update_serving_endpoint_with_retry(self, max_retries: int = 10, retry_interval: int = 30) -> None:
        """Deploy or update a serving endpoint with retry mechanism.

        :param max_retries: Maximum number of retry attempts
        :param retry_interval: Time interval between retries in seconds
        """
        super().deploy_or_update_serving_endpoint_with_retry(
            served_entities=self.served_entities, max_retries=max_retries, retry_interval=retry_interval
        )

    def delete_feature_spec(self) -> None:
        """Delete the feature specification.

        Deletes the feature specification using the FeatureEngineeringClient.
        """
        self.fe.delete_feature_spec(name=self.feature_spec_name)


class FeatureLookupServing(ModelServing):
    """Manages feature lookup serving for a model endpoint.

    This class handles the creation and management of online tables for feature lookup,
    as well as the deployment of serving endpoints with retry mechanisms.

    Attributes:
        feature_table_name (str): Name of the feature table.

    """

    def __init__(
        self,
        endpoint_name: str,
        model_name: str,
        feature_table_name: str,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
    ) -> None:
        """Initialize a FeatureLookupServing instance.

        :param model_name: Name of the model to be served
        :param endpoint_name: Name of the serving endpoint
        :param feature_table_name: Name of the feature table
        :param workload_size: Size of the workload, defaults to "Small"
        :param scale_to_zero: Whether to scale to zero, defaults to True
        """
        super().__init__(
            endpoint_name=endpoint_name,
            model_name=model_name,
            version=version,
            workload_size=workload_size,
            scale_to_zero=scale_to_zero,
        )
        self.feature_table_name = feature_table_name

    def create_online_table(self) -> None:
        """Create an online table in Databricks.

        This method creates an online table using the specified feature table as the source.
        The online table is configured with a triggered scheduling policy and without performing a full copy.

        :param self: The instance of the class containing this method.
        """
        FeatureServing.form_online_table(feature_table_name=self.feature_table_name, primary_keys=["booking_id"])

    def deploy_or_update_serving_endpoint_with_retry(self, max_retries: int = 10, retry_interval: int = 30) -> None:
        """Deploy or update a serving endpoint with retry mechanism.

        :param max_retries: Maximum number of retry attempts
        :param retry_interval: Time interval between retries in seconds
        """
        super().deploy_or_update_serving_endpoint_with_retry(max_retries=max_retries, retry_interval=retry_interval)

    def update_online_table(self, config: Config) -> None:
        """Update the online table using the specified pipeline configuration.

        This function starts a pipeline update and monitors its progress until completion or failure.

        :param config: Configuration object containing pipeline details.
        :raises SystemError: If the online table update fails.
        """
        update_response = self.workspace.pipelines.start_update(pipeline_id=config.pipeline_id, full_refresh=False)

        while True:
            update_info = self.workspace.pipelines.get_update(
                pipeline_id=config.pipeline_id, update_id=update_response.update_id
            )
            state = update_info.update.state.value

            if state == "COMPLETED":
                logger.info("Online table update completed successfully.")
                break
            elif state in ["FAILED", "CANCELED"]:
                logger.error("Pipeline update failed")
                raise SystemError("Online table update failed.")
            elif state == "WAITING_FOR_RESOURCES":
                logger.warning(f"Pipeline is in {state}.")
            else:
                logger.info(f"Pipeline is in {state}.")

            time.sleep(30)
