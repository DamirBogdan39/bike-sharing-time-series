from dataclasses import dataclass


@dataclass()
class ModelHparamsParams:
    max_depth: int
    n_estimators: int
    num_leaves: int
