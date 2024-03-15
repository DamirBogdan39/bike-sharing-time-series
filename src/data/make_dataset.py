from typing import Tuple

import pandas as pd

from src.entities import SplittingParams


def split_train_test_data(
        data: pd.DataFrame,
        params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data["dteday"] = pd.to_datetime(data["dteday"])
    train_data = data[data['dteday'] <= params.split_date]
    test_data = data[data['dteday'] > params.split_date]

    return train_data, test_data

def write_train_test_data(raw_path,train_path, test_path):

    df = pd.read_csv(raw_path)
    train, test = split_train_test_data(df, SplittingParams)

    train.to_csv(train_path)
    test.to_csv(test_path)

    return
