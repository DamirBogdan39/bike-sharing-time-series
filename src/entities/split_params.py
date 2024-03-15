from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    split_date: str = field(default="2012-11-30")