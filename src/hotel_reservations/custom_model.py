"""Custom Model Module."""

from typing import Union

import cloudpickle
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

from hotel_reservations.config import Config, Tags
from tests.consts import PROJECT_DIR

# model_file, model_path, model_file_path
# https://www.perplexity.ai/search/you-are-an-expert-on-mlflow-wi-qG83lj2oSISJCB12sMsHKQ


def load_model(file_path: str) -> object:
    """Load a serialized model from a file using cloudpickle.

    :param file_path: The path to the file containing the serialized model.
    :return: The deserialized model object.
    """
    with open(file_path, "rb") as model_file:
        model = cloudpickle.load(model_file)
    return model


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """A wrapper class for machine learning models to be used with MLflow.

    This class encapsulates a model and provides a standardized predict method.
    """

    def __init__(self, model: object) -> None:
        self.model = model

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Union[pd.DataFrame, np.array],  #  noqa
    ) -> Union[pd.DataFrame, np.array]:  #  noqa
        """Perform predictions using the wrapped model.

        :param context: The MLflow PythonModelContext, which provides runtime information.
        :param model_input: The input data for prediction, either as a pandas DataFrame or a NumPy array.
        :return: The prediction results as a pandas DataFrame or a NumPy array.
        """
        return self.model.predict(model_input)


class CustomModel:
    """Custom model."""

    def __init__(self, config: Config, tags: Tags, model: object, code_paths: list[str]) -> None:
        self.config = config
        self.tags = tags
        self.model = ModelWrapper(model)
        self.code_paths = code_paths

        # initialize from config
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_custom
        self.model_name = self.config.model.name
        self.model_artifact_path = self.config.model.artifact_path

        # disable autologging in order not to have multiple experiments and runs :)
        mlflow.autolog(disable=True)

    def log_model(self) -> None:
        """Log model."""
        mlflow.set_experiment(self.experiment_name)

        additional_pip_deps = ["pyspark==3.5.0"]
        if self.code_paths:
            for package in self.code_paths:
                whl_name = package.split("/")[-1]
                additional_pip_deps.append(f"code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            #  work around
            TRAIN_TEST_PRED_FOLDER = PROJECT_DIR / "tests" / "test_data" / "train_test_pred"
            xtrain = pd.read_csv((TRAIN_TEST_PRED_FOLDER / "xtrain.csv").resolve().as_posix())
            ypred = pd.read_csv((TRAIN_TEST_PRED_FOLDER / "ypred.csv").resolve().as_posix())

            # Log the model
            signature = infer_signature(model_input=xtrain, model_output=ypred)

            # dataset = mlflow.data.  #  noqa
            dataset = mlflow.data.from_pandas(xtrain, (TRAIN_TEST_PRED_FOLDER / "xtrain.csv").resolve().as_posix())
            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            mlflow.pyfunc.log_model(
                python_model=self.model,
                artifact_path=self.model_artifact_path,
                code_path=self.code_paths,
                conda_env=conda_env,
                signature=signature,
                input_example=xtrain.head(4),
            )

        logger.info(f"✅ Model logged successfully to {self.model_artifact_path}.")

    def register_model(self) -> None:
        """Register."""
        raise NotImplementedError()

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve MLflow run dataset."""
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        return dataset_source.load()
        logger.info("✅ Dataset source loaded.")

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata."""
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        return metrics, params
        logger.info("✅ Dataset metadata loaded.")

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> Union[pd.DataFrame, np.array]:  #  noqa
        """Load latest."""
        raise NotImplementedError()
