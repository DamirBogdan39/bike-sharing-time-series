from dataclasses import dataclass
from typing import List


@dataclass()
class FeatureParams:
    feature_names: List[str]
    period: List[int]
    lag_target: str
    lag: int