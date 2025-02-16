"""Serving Module."""

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

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
        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(self, served_entities: list[ServedEntityInput]) -> None:
        """Deploy or update the serving endpoint with the given served entities.

        :param served_entities: List of ServedEntityInput objects to be deployed or updated
        """
        super().deploy_or_update_serving_endpoint(self.served_entities)
