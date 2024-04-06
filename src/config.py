import pathlib

import pydantic

MAX_DOWNLOAD_TIME = 0.2

IMAGE_DOWNLOAD_PATH = pathlib.Path("/tmp/images")


class DataConfig(pydantic.BaseModel):
    buffer_size: int = 1000
    data_len: int = 100
    train_len: int = 90
    small_dataset: str = "laion/220k-gpt4vision-captions-from-livis"
    large_dataset: str = "laion/laion400m"
    dataset: str = small_dataset


class ModelConfig(pydantic.BaseModel):
    text_model: str = "microsoft/xtremedistil-l6-h256-uncased"  # 51 mb
    vision_model: str = "edgenext_small"  # 20 mb
    projection_layers: int = 3
    embed_dim: int = 256
    transformer_embed_dim: int = 768
    max_len: int = 128  # 77
    cls_type: bool = True
    freeze_vision_base: bool = False
    freeze_text_base: bool = False


class TrainerConfig(pydantic.BaseModel):
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 5e-4
    accumulate_grad_batches: int = 1
    temperature: float = 1.0
    vision_freeze_layers: int = 2
    lambda_1: float = 1.0
    lambda_2: float = 1.0

    val_check_interval: int = 1000

    run_openai_clip: bool = False

    _model_config: ModelConfig = ModelConfig()
    _data_config: DataConfig = DataConfig()
