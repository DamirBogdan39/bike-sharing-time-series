from omegaconf import DictConfig
import hydra
import os
import logging

from src.data import write_train_test_data

# Setup the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("create_datasets")

@hydra.main(config_path="../conf/path_config", config_name="path_config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting script")
    os.chdir(hydra.utils.to_absolute_path("."))
    logger.info(f"Read data path: {cfg.raw_data_path}")
    logger.info(f"Train data path: {cfg.train_data_path}")
    logger.info(f"Test data path: {cfg.test_data_path}")
    write_train_test_data(cfg.raw_data_path, cfg.train_data_path, cfg.test_data_path)
    logger.info("Data written successfully.")

if __name__ == "__main__":
    main()
    