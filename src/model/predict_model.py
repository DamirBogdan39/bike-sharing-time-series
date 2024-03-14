from typing import Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def make_prediction(model: LGBMRegressor, features: pd.DataFrame) -> np.ndarray:
    prediction = model.predict(features)
    prediction = np.around(prediction).astype(int)
    return prediction


def evaluate_model(prediction: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "mean_absolute_error": mean_absolute_error(target, prediction),
        "root_mean_squared_error": np.sqrt(mean_squared_error(target, prediction)),
        "r2_score": r2_score(target, prediction),
    }
