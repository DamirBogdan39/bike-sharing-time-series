from lightgbm import LGBMRegressor

from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV
)
from src.utils import read_data
from src.entities import OptimizerPipelineParams
from src.features import preprocess, get_features_and_target


def optimize(optimizer_pipeline_params: OptimizerPipelineParams):
    """
    An optimization function which performs GridSearchCV over different hyperparameters.
    Reads in the data, preprocesses the dataframe and performs GridSearchCV to find the optimal hyperparameters.

    Parameters:
    - optmizer_pipeline_params (OptimizerPipelineParams): OptimizerPipelineParams class object.
    """
    df = read_data(optimizer_pipeline_params.path_config.train_data_path)
    
    df = preprocess(df, optimizer_pipeline_params.feature_params)

    X_train, y_train = get_features_and_target(df)

    lgb = LGBMRegressor(verbose=-1, random_state=42) 
    tss = TimeSeriesSplit(n_splits=5, test_size=24*30, gap=24)

    param_grid = {"max_depth": optimizer_pipeline_params.param_grid.max_depth,
                  "n_estimators": optimizer_pipeline_params.param_grid.n_estimators,
                  "num_leaves": optimizer_pipeline_params.param_grid.num_leaves}
    
    gscv = GridSearchCV(estimator=lgb, param_grid=param_grid, cv=tss, scoring="neg_mean_absolute_error", verbose=10)

    gscv.fit(X_train, y_train)

    best_hparams = gscv.best_params_
    return best_hparams
