from omegaconf import DictConfig
import hydra
import os
import logging
import yaml

from src.entities import OptimizerPipelineParamsSchema
from src.model import optimize

# Setup the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_optimization")

# Define main function
@hydra.main(config_path="../conf", config_name="optimizer_conf")
def main(cfg: DictConfig) -> None:
    logger.info("Starting script")
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = OptimizerPipelineParamsSchema()
    params = schema.load(cfg)
    best_hparams = optimize(params)

    logger.info(f"Best hyperparameters: {best_hparams}")
    logger.info(f"Writing best hyperparameters to: {params.path_config.model_hparams_path}")
    with open(params.path_config.model_hparams_path, "w") as file:
        yaml.dump(best_hparams, file, default_flow_style=False)
    logger.info("Finished script")

if __name__ == "__main__":
    main()
