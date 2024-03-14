from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    test_size: int = field(default=1*24*31)
    random_state: int = field(default=42)
    shuffle: bool = field(default=False)