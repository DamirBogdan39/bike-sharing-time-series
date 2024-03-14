import unittest
import pandas as pd
import numpy as np
from typing import Dict
from src.model.predict_model import evaluate_model

from src.features.build_features import trig_transformer

class TestTrigTransformer(unittest.TestCase):

    def test_trig_transformer(self):
        df = pd.DataFrame({
            "column_to_transform": [0, 1, 2, 3, 4],
            "extra_column": ["a", "b", "c", "d", "e"]
            })
        column = "column_to_transform"
        period = 2

        result = trig_transformer(df, column, period)

        df_expected = pd.DataFrame({
            "column_to_transform": [0, 1, 2, 3, 4],
            "extra_column": ["a", "b", "c", "d", "e"],
            "column_to_transform_sin": [np.sin(x / period * 2 * np.pi) for x in df[column]],
            "column_to_transform_cos": [np.cos(x / period * 2 * np.pi) for x in df[column]],})

        pd.testing.assert_frame_equal(result, df_expected)


class TestEvaluateModel(unittest.TestCase):
    
    def test_evaluate_model(self):
        prediction = np.array([2, 2, 2])
        target = pd.Series([3, 3, 3])
        
        expected_output = {
            "mean_absolute_error": 1.0, 
            "root_mean_squared_error": 1.0,
            "r2_score": 0.0
        }
        
        self.assertTrue(np.allclose(list(evaluate_model(prediction, target).values()),
                                    list(expected_output.values()), 
                                    atol=1e-5))
    
        prediction = np.array([3, 3, 3])
        target = pd.Series([3, 3, 3])
        
        expected_output = {
            "mean_absolute_error": 0.0, 
            "root_mean_squared_error": 0.0,
            "r2_score": 1.0
        }
        
        self.assertTrue(np.allclose(list(evaluate_model(prediction, target).values()),
                                    list(expected_output.values()), 
                                    atol=1e-5))
    
    
if __name__ == "__main__":
    unittest.main()
