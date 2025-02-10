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

#### Basic Model

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
