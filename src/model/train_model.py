from lightgbm import LGBMRegressor
import pandas as pd
from src.entities import TrainingPipelineParams

def train_model(X: pd.DataFrame, y: pd.Series, training_pipeline_params: TrainingPipelineParams):
    """
    A training function to load the model retrain in on the training data with best hyperparameters gotten from the model_params_optmizier.py script.

    Parameters:
    - X: pd.DataFrame: features
    - y: pd.Series: target
    - training_pipeline_params (TrainingPipelineParams): TrainingPipelineParams class object.

    Returns:
    - A retrained LGBMRegressor object to perform prediction.
    """
    lgb = LGBMRegressor(verbose=-1,
                        random_state=42,
                        max_depth=training_pipeline_params.model_hparams.max_depth,
                        n_estimators=training_pipeline_params.model_hparams.n_estimators,
                        num_leaves=training_pipeline_params.model_hparams.num_leaves,
                        )
    
    lgb.fit(X, y)

    return lgb