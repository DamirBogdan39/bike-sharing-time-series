from sklearn.preprocessing import FunctionTransformer, SplineTransformer
import numpy as np
import pandas as pd
from typing import List

from src.entities import FeatureParams

def trig_transformer(df: pd.DataFrame, column: str, period:int) -> pd.DataFrame:
    """
    Takes a DataFrame, applies sin and cos transformation to a given column, 
    adds the transformed column to the DataFrame and returns the DataFrame.

    Parameters:
    - df (pd.DataFrame): Original DataFrame
    - column (str): Name of the column in the DataFrame to be transformed.
    - period (int): The period to use in the transformation

    Returns:
    - df (pd.DataFrame): DataFrame which includes transformed column.
    """

    sin_trans = FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))
    cos_trans = FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    sin_transf = sin_trans.fit_transform(df[[column]].values)
    cos_transf = cos_trans.fit_transform(df[[column]].values)

    sin_series = pd.Series(sin_transf.flatten(), name=f"{column}_sin", index=df.index)
    cos_series = pd.Series(cos_transf.flatten(), name=f"{column}_cos", index=df.index)

    return pd.concat([df, sin_series, cos_series], axis=1)


def periodic_spline_transformer(period: int, n_splines=None, degree=3) -> SplineTransformer:
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1
    
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


def spline_transformer(df: pd.DataFrame, column_name: str, period: int) -> pd.DataFrame:
    """
    Takes a DataFrame, applies Spline Tranformer to a given column, 
    adds the transformed column to the DataFrame and returns the DataFrame.

    Parameters:
    - df (pd.DataFrame): Original DataFrame
    - column (str): Name of the column in the DataFrame to be transformed.
    - period (int): Period parameter for the Spline Transformer.

    Returns:
    - df (pd.DataFrame): DataFrame which includes transformed column.
    """


    splines = periodic_spline_transformer(period, n_splines=None).fit_transform(df[[column_name]])
    splines_df = pd.DataFrame(
        splines,
        columns=[f"{column_name}_spline_{i}" for i in range(splines.shape[1])],
        index=df.index
    )

    return pd.concat([df, splines_df], axis=1)


def get_lag_features(df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
    """
    Create lag features from the target column. Lags are created from previous values of target.

    Parameters:
    - df (pd.DataFrame): Original DataFrame
    - column (str): Name of the target variable.
    - lags (list): List of integers indicating which lag features to create. 
                    e.g. [1, 7, 30] will create lags of 1, 7 and 30 previous hours.

    Returns:
    - df (pd.DataFrame): DataFrame which includes lag columns.
    """
    df["dteday"] = pd.to_datetime(df["dteday"])
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)

    return df


def preprocess(df: pd.DataFrame,
               params: FeatureParams) -> pd.DataFrame:
    """
    Takes in a raw dataframe, performs feature engineering and returns a clean dataframe prepared for training/testing.

    Parameters:
    - df (pd.DataFrame): Raw dataframe.

    Returns:
    - df (pd.DataFrame): Clean dataframe with engineered features.
    """
    for s, n in zip(params.feature_names, params.period):
        df = trig_transformer(df, s, n)
        df = spline_transformer(df, s, n)

    df = get_lag_features(df, params.lag_target, params.selected_lags)

    feats = (["season", "yr", "mnth", "holiday", "weekday", "workingday"] +
         ["hr_sin", "hr_cos", "mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos"] +
         [f"hr_spline_{i}" for i in range(24)] +
         [f"mnth_spline_{i}" for i in range(12)] +
         [f"weekday_spline_{i}" for i in range(7)] +
         [f"cnt_lag_{i}" for i in params.selected_lags])
    
    df = df[feats + ["cnt"]]

    return df


def get_features_and_target(df: pd.DataFrame) -> pd.Series:
    return df.drop("cnt", axis=1), df["cnt"]
