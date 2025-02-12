<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks course

## Practical information
- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Code for the lecture is shared before the lecture.
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset.
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the 25th of November.


## Set up your environment
In this course, we use Databricks 15.4 LTS runtime, which uses Python 3.11.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv venv -p 3.11 venv
source venv/bin/activate
uv pip install -r pyproject.toml --all-extras
uv lock
```

## Modules
### Data Ingestion (DataLoader)
 `DataLoader` class is designed for cleaning and processing hotel reservation data using PySpark and pandas. Here's a summary of its key features and functionality:

1. Data Loading: The class can load data from CSV files, supporting both local file systems and Databricks environments.

2. Data Validation: It includes methods to validate required columns, data types, and check for null values in the dataset.

3. Data Preprocessing: The class offers various data cleaning operations such as:
   - Renaming columns based on configuration
   - Converting and validation column data types
   - Applying value corrections (e.g., mapping categorical values to numeric ones)
   - Normalizing arrival dates
   - Performing final quality checks on the data

4. Data Splitting: It provides functionality to split the data into training and testing sets, as well as extracting non-online market segment data.

5. Data Saving: For Databricks environments, it includes methods to save processed data to a catalog, creating tables with timestamps and enabling change data feed.

6. Logging and Error Handling: The code extensively uses logging to track the data processing steps and implements error handling for various scenarios.

7. Configuration: The class uses a configuration object to manage settings for features, target variables, and data types.

Overall, it is a comprehensive data preparation pipeline for hotel reservation data, emphasizing data quality, flexibility, and compatibility with both local and cloud-based (Databricks) environments.

### Model Training & Registration

#### `Basic Model`

*   The `BasicModel` class provides a foundational framework for building and deploying machine learning models, specifically tailored for the Databricks environment using MLflow for experiment tracking, model management, and deployment. It encapsulates the essential steps of a machine learning workflow, from data loading and preprocessing to model training, evaluation, and registration within Unity Catalog (UC).

*   **Key Features:**

    *   **Data Loading:** Facilitates loading training and testing datasets directly from Databricks tables, leveraging Spark for efficient data handling and conversion to Pandas DataFrames.
    *   **Feature Engineering:** Incorporates a preprocessing pipeline using `scikit-learn's` `ColumnTransformer` and `OneHotEncoder` to handle categorical features effectively, ensuring compatibility with machine learning algorithms.
    *   **Model Training:** Supports training of `LGBMClassifier` models (or any `scikit-learn`-compatible model) using a defined pipeline, streamlining the training process.
    *   **MLflow Integration:** Seamlessly integrates with MLflow to log model parameters, metrics (accuracy, AUC, classification reports), and artifacts, enabling comprehensive experiment tracking and reproducibility. It also logs the training dataset as an MLflow input, capturing its name and version.
    *   **Model Registration:** Automates the registration of trained models in Unity Catalog, assigning aliases (e.g., "latest-model") for easy retrieval and deployment of the most recent version.
    *   **Model Deployment:** Includes functionality to load the latest registered model from MLflow and generate predictions on new data, facilitating model serving and inference.
    *   **Configuration-Driven:** Designed to be highly configurable through a `Config` class, allowing users to specify model parameters, feature lists, target variables, and MLflow settings via configuration files.
    *   **Databricks-Centric:** Optimized for use within Databricks environments, with specific checks and utilities (e.g., `is_databricks()`, `get_delta_table_version()`, `get_current_git_sha()`) to ensure compatibility and leverage Databricks features.

*   **Workflow:** The module orchestrates a typical machine learning workflow: loading data from Databricks tables, preprocessing features, training a model, logging model metadata and performance to MLflow, registering the model in Unity Catalog, and providing a means to load and apply the latest registered model for inference.

*   **Use Case:** Ideal for projects requiring a standardized and repeatable machine learning pipeline within Databricks, with a focus on experiment tracking, model governance, and simplified deployment through MLflow and Unity Catalog.

#### `Custom Model` 

*   The `CustomModel` module provides a flexible framework for integrating arbitrary pre-trained machine learning models into an MLflow workflow, particularly within a Databricks environment, leveraging Unity Catalog (UC) for model governance. It's designed to handle models that might not be directly supported by MLflow's built-in logging functions, allowing you to wrap and deploy models created using any library.

*   **Key Features:**

    *   **Model Wrapping:** Employs a `ModelWrapper` class that extends `mlflow.pyfunc.PythonModel`, providing a standardized `predict` interface for any underlying model, regardless of its original framework.  This allows MLflow to treat the custom model as a standard Python function.
    *   **Serialization:**  Uses `cloudpickle` for serializing and deserializing the underlying model.  This allows for handling a wider variety of model types than standard pickle.
    *   **Code Dependency Management:**  Handles code dependencies by allowing you to specify a list of `code_paths` (e.g., paths to `.whl` files). These dependencies are included in the MLflow environment, ensuring that the model can be loaded and executed correctly.
    *   **MLflow Integration:** Seamlessly integrates with MLflow to log the wrapped model, including its signature, input examples, and environment.  It leverages `mlflow.pyfunc.log_model` for this.
    *   **Model Registration:** Automates the registration of trained models in Unity Catalog, assigning aliases (e.g., "latest-model") for easy retrieval and deployment of the most recent version.
    *   **Model Deployment:** Includes functionality to load the latest registered model from MLflow and generate predictions on new data, facilitating model serving and inference.
    *   **Configuration-Driven:** Designed to be highly configurable through a `Config` class, allowing users to specify model parameters, feature lists, target variables, and MLflow settings via configuration files.
    *   **Custom Environment:**  Creates a custom Conda environment for the model, including necessary dependencies like `pyspark` and any additional packages specified through `code_paths`.
    *   **Flexibility:** Enables the deployment of models trained using any Python-compatible machine learning library, overcoming limitations of MLflow's autologging for specific frameworks.

*   **Workflow:** The module encapsulates the process of wrapping a pre-trained model, defining its dependencies, logging it to MLflow, registering it in Unity Catalog, and deploying it for inference.  It allows you to bring your own model and integrate it into a managed MLflow environment.

*   **Use Case:** Ideal for scenarios where you have a pre-trained model (e.g., a model trained outside of Databricks or using a less common library) that you want to deploy and manage within the Databricks ecosystem using MLflow and Unity Catalog.  It provides a way to standardize the deployment process for diverse model types.

####  `FeatureLookUpModel model`

*   The `FeatureLookUpModel` class provides an advanced approach to machine learning for hotel reservation prediction by leveraging Databricks Feature Engineering Client for feature creation and management, in addition to MLflow for experiment tracking and model deployment. This class uses Feature Engineering to create and manage features used by the model.
*   **Key Features:**

    *   **Feature Engineering with Databricks Feature Engineering Client:** Uses the Databricks Feature Engineering Client to create and manage feature tables and functions within the Databricks environment.
    *   **Feature Lookup:** Utilizes `FeatureLookup` to dynamically enrich training data with features from managed feature tables, ensuring data consistency and reducing feature engineering boilerplate.
    *   **Automated Feature Table Creation:** Automatically creates and populates feature tables (e.g., `hotel_features`) within Unity Catalog, simplifying the feature engineering process.
    *   **Feature Function Definition:** Enables the creation and registration of custom feature functions (e.g., calculating lead time) as SQL functions within Databricks, promoting code reuse and simplifying feature transformations.
    *   **MLflow Integration:** Integrates with MLflow to log model parameters, metrics (accuracy, AUC, classification reports), and the trained model, ensuring comprehensive experiment tracking and reproducibility.
    *   **Model Training:** Trains an `LGBMClassifier` model (configurable via `config.parameters`) using a `scikit-learn` Pipeline, simplifying the training process.
    *   **Model Registration:** Registers the trained model in Unity Catalog, enabling versioning and easy deployment, similar to the `BasicModel`.
    *   **Databricks-Centric:** Designed specifically for Databricks environments, leveraging the Feature Engineering Client, Spark for data handling, and Unity Catalog for model and feature governance.
*   **Workflow:**  The module orchestrates a feature-rich machine learning pipeline: creating feature tables and functions, loading data and enriching it with Feature Lookups, training an LGBMClassifier model, logging metadata and performance to MLflow, and registering the model in Unity Catalog.
*   **Use Case:**  Well-suited for projects that require advanced feature engineering capabilities, such as feature sharing, feature reuse, and automated feature management within Databricks. It's ideal for scenarios where feature tables are managed separately and need to be dynamically joined with training data.

