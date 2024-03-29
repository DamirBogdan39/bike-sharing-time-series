import json
import pickle
from typing import NoReturn

import pandas as pd


def read_data(path: str) -> pd.DataFrame:
    """
    Reads the .csv file and returns a dataframe for further processing/analysis.

    Parameters:
    path (str): a path to the .csv file.

    Returns:
    - df (pd.DataFrame): pandas DataFrame object.
    """
    data = pd.read_csv(path)
    return data


def save_metrics_to_json(file_path: str, metrics: dict) -> NoReturn:
    with open(file_path, "w") as metric_file:
        json.dump(metrics, metric_file)


def save_pkl_file(input_file, output_name: str) -> NoReturn:
    with open(output_name, "wb") as f:
        pickle.dump(input_file, f)


def load_pkl_file(input_: str):
    with open(input_, "rb") as fin:
        res = pickle.load(fin)
    return res
