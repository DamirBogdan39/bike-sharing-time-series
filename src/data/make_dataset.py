from typing import Tuple

import pandas as pd

from sklearn.model_selection import train_test_split
from src.entities import SplittingParams


def split_train_test_data(
        data: pd.DataFrame,
        params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_data, test_data = train_test_split(
        data, test_size=params.test_size, random_state=params.random_state, shuffle=params.shuffle
    )
    return train_data, test_data

def write_train_test_data(raw_path,train_path, test_path):

    df = pd.read_csv(raw_path)
    train, test = split_train_test_data(df, SplittingParams)

    train.to_csv(train_path)
    test.to_csv(test_path)

    return