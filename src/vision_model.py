import timm
from timm import data
import torch.nn as nn
from torchvision import transforms

from src.config import TinyCLIPVisionConfig


def get_vision_base(
    config: TinyCLIPVisionConfig,
) -> tuple[nn.Module, int]:
    base = timm.create_model(config.vision_model, num_classes=0, pretrained=True)
    num_features = base.num_features
    return base, num_features


def get_vision_transform(config: TinyCLIPVisionConfig) -> transforms.Compose:
    timm_config = data.resolve_data_config({}, model=config.vision_model)
    transform = data.transforms_factory.create_transform(**timm_config)
    return transform  # type: ignore
