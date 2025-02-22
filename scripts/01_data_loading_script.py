"""Data loading and preprocessing script."""

import os
import pathlib

from dotenv import load_dotenv
from loguru import logger

from hotel_reservations.config import Config
from hotel_reservations.data_ingestion import DataFabricator, DataLoader
from hotel_reservations.utility import create_parser, is_databricks, setup_logging

# setup
envfile_path = pathlib.Path().joinpath("../project.env").resolve().as_posix()
print(f"{envfile_path =}")

load_dotenv(envfile_path)

INGESTION_LOGS = os.environ["INGESTION_LOGS"]
INGESTION_LOGS = pathlib.Path(INGESTION_LOGS).resolve().as_posix()
print(f"{INGESTION_LOGS = }")

setup_logging(INGESTION_LOGS)

if is_databricks():
    DATABRICKS_FILE_PATH = os.environ["DATA_FILEPATH_DATABRICKS"]
    CONFIG_FILE_PATH = pathlib.Path("../project_config.yml").resolve().as_posix()
    DATA_VOLUME = pathlib.Path(DATABRICKS_FILE_PATH).parent

print(f"{DATABRICKS_FILE_PATH = }")
print(f"{CONFIG_FILE_PATH = }")
print(f"{DATA_VOLUME.as_posix() = }")


args = create_parser()
print(f"{args.root_path}")
print(f"{args.env}")

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = Config.from_yaml(config_file=config_path, env=args.env)

# generate synthetic data by mimicking real data flow behavior
dataloader = DataLoader(filepath=DATABRICKS_FILE_PATH, config=config)
data_factory = DataFabricator(payload=dataloader)
synthetic_df = data_factory.synthesize(num_rows=100)
logger.info("Synthetic data generated")

output_filename = (DATA_VOLUME / "synthetic.csv").as_posix()
data_factory.to_csv(dataframe=synthetic_df, output_filename=output_filename)
logger.info("-" * 50)

# initialize data ingestion
dataloader = DataLoader(filepath=output_filename, config=config)

# Preprocess the data
dataloader.process_data()

# Split the data
train_set, test_set = dataloader.split_data()

# Save to catalog
dataloader.save_to_catalog(train_set=train_set, test_set=test_set)
