from dataclasses import dataclass
from .feature_params import FeatureParams
from .path_params import PathParams
from marshmallow_dataclass import class_schema


@dataclass()
class PredictPipelineParams:
    path_config: PathParams
    feature_params: FeatureParams


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)
