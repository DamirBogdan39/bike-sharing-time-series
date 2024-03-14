from dataclasses import dataclass
from .feature_params import FeatureParams
from .path_params import PathParams
from .param_grid import ParamGridParams
from marshmallow_dataclass import class_schema


@dataclass()
class OptimizerPipelineParams:
    path_config: PathParams
    param_grid: ParamGridParams
    feature_params: FeatureParams

OptimizerPipelineParamsSchema = class_schema(OptimizerPipelineParams)
