from dataclasses import dataclass
from typing import List

@dataclass
class ParamGridParams:
    max_depth: List[int]
    n_estimators: List[int]
    num_leaves: List[int]