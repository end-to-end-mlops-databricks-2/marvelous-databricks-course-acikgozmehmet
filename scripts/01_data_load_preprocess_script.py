"""Data loading, preprocessing, and synthetic data generation script."""

import os
import pathlib

from dotenv import load_dotenv
from loguru import logger

from hotel_reservations.config import Config
from hotel_reservations.data_ingestion import DataFabricator, DataLoader
from hotel_reservations.utility import create_parser, is_databricks, setup_logging

# Setup environment and configurations
envfile_path = pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f"Loading environment variables from: {envfile_path}")

load_dotenv(envfile_path)

INGESTION_LOGS = os.environ["INGESTION_LOGS"]
INGESTION_LOGS = pathlib.Path(INGESTION_LOGS).resolve().as_posix()
print(f"Ingestion logs will be stored at: {INGESTION_LOGS}")

setup_logging(INGESTION_LOGS)

if is_databricks():
    DATABRICKS_FILE_PATH = os.environ["DATA_FILEPATH_DATABRICKS"]
    DATA_VOLUME = pathlib.Path(DATABRICKS_FILE_PATH).parent

logger.info(f"Databricks file path: {DATABRICKS_FILE_PATH}")
logger.info(f"Data volume: {DATA_VOLUME.as_posix()}")


# Parse command-line arguments
args = create_parser()
logger.info(f"Root path: {args.root_path}")
logger.info(f"Environment: {args.env}")

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

# Load configuration
config = Config.from_yaml(config_file=config_path, env=args.env)

# Generate synthetic data
dataloader = DataLoader(filepath=DATABRICKS_FILE_PATH, config=config)
data_factory = DataFabricator(payload=dataloader)
synthetic_df = data_factory.synthesize(num_rows=100)
logger.info("Synthetic data generated successfully")

output_filename = (DATA_VOLUME / "synthetic.csv").as_posix()
data_factory.to_csv(dataframe=synthetic_df, output_filename=output_filename)
logger.info(f"Synthetic data saved to: {output_filename}")
logger.info("-" * 50)

# Initialize data ingestion
dataloader = DataLoader(filepath=output_filename, config=config)
logger.info("Data loader initialized with synthetic data")

# Preprocess the data
dataloader.process_data()
logger.info("Data preprocessing completed")

# Split the data into train and test sets
train_set, test_set = dataloader.split_data()
logger.info("Data split into train and test sets")

# Save processed data to catalog
dataloader.save_to_catalog(train_set=train_set, test_set=test_set)
logger.info("Processed data saved to catalog")
