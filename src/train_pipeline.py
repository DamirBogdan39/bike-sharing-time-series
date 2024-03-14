from omegaconf import DictConfig
import hydra
import os
import logging

from src.entities import TrainingPipelineParams, TrainingPipelineParamsSchema
from src.model import train_model, make_prediction, evaluate_model
from src.features import preprocess, get_features_and_target                          
from src.utils import read_data, save_metrics_to_json, save_pkl_file

# Setup the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_training")

def train_pipeline(params: TrainingPipelineParams):
    
    logger.info("Loading training data.")
    df = read_data(params.path_config.train_data_path)
    df = preprocess(df, params.feature_params)
    X_train, y_train = get_features_and_target(df)
    logger.info("Training data loaded.")

    logger.info("Training model.")
    lgb = train_model(X_train, y_train, params)
    logger.info("Finished training.")

    logger.info("Loading testing data.")
    df_test = read_data(params.path_config.test_data_path)
    df_test = preprocess(df_test, params.feature_params)
    X_test, y_test = get_features_and_target(df_test)
    logger.info("Testing data loaded.")

    logger.info("Making prediction.")
    prediction = make_prediction(lgb, X_test)
    logger.info("Evaluating model.")
    metrics = evaluate_model(prediction, y_test)
    logger.info(f"Metrics for test dataset is {metrics}")
    logger.info("Saving model.")
    save_pkl_file(lgb, params.path_config.output_model_path)
    logger.info("Model saved.")
    save_metrics_to_json(params.path_config.metrics_path, metrics)
    logger.info(f"Metrics saved to {params.path_config.metrics_path}.")

    return metrics

# Define main function
@hydra.main(config_path="../conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    logger.info("Starting script")
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    train_pipeline(params)
    logger.info("Finished script")

if __name__ == "__main__":
    main()
    import os
    print(os.getcwd())

