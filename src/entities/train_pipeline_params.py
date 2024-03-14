from dataclasses import dataclass
from .feature_params import FeatureParams
from .path_params import PathParams
from .param_grid import ParamGridParams
from .model_hparams import ModelHparamsParams

from marshmallow_dataclass import class_schema


@dataclass()
class TrainingPipelineParams:
    path_config: PathParams
    feature_params: FeatureParams
    model_hparams: ModelHparamsParams

TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)
