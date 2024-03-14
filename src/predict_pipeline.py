import os
import logging.config

import pandas as pd
from omegaconf import DictConfig
import hydra

from src.entities.predict_pipeline_params import PredictPipelineParams, \
    PredictPipelineParamsSchema
from src.model import make_prediction
from src.features import preprocess, get_features_and_target                          
from src.utils import read_data, load_pkl_file

logger = logging.getLogger("predict_pipeline")


def predict_pipeline(params: PredictPipelineParams):
    
    logger.info("Loading pretrained model.")
    lgb = load_pkl_file(params.path_config.output_model_path)    
    logger.info("Model loaded.")

    logger.info("Loading data.")
    df_test = read_data(params.path_config.test_data_path)
    df_test = preprocess(df_test, params.feature_params)
    X_test, _ = get_features_and_target(df_test)
    logger.info("Data loaded.")

    logger.info("Making prediction.")
    prediction = make_prediction(lgb, X_test)

    df_prediction = pd.DataFrame(prediction)

    df_prediction.to_csv(params.path_config.output_data_path, header=False)

    logger.info(
        f"Prediction is done and saved to the file {params.path_config.output_data_path}"
    )
    return df_prediction


    
@hydra.main(config_path="../conf", config_name="predict_config")
def predict_pipeline_start(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = PredictPipelineParamsSchema()
    params = schema.load(cfg)
    print(os.getcwd())
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_start()
