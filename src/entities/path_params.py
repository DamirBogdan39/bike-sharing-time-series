from dataclasses import dataclass


@dataclass
class PathParams:
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    output_model_path: str
    model_hparams_path: str
    metrics_path: str
    output_data_path: str