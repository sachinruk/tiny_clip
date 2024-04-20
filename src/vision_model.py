import timm
from timm import data

from src import config


def get_vision_base_and_transform(config: config.TrainerConfig):
    base = timm.create_model(config._model_config.vision_model, num_classes=0)
    timm_config = data.resolve_data_config({}, model=base)
    transform = data.transforms_factory.create_transform(**timm_config)
    return base, transform
