import pathlib

import pydantic
from transformers import PretrainedConfig

MAX_DOWNLOAD_TIME = 0.2

IMAGE_DOWNLOAD_PATH = pathlib.Path("./data/images")
WANDB_LOG_PATH = pathlib.Path("/tmp/wandb_logs")

IMAGE_DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
WANDB_LOG_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "tiny_clip"

WANDB_ENTITY = "sachinruk"


class DataConfig(pydantic.BaseModel):
    buffer_size: int = 1000
    data_len: int = 100
    train_len: int = 90
    small_dataset: str = "laion/220k-gpt4vision-captions-from-livis"
    large_dataset: str = "laion/laion400m"
    dataset: str = small_dataset


class TinyCLIPTextConfig(PretrainedConfig):
    model_type = "text"

    def __init__(
        self,
        text_model: str = "microsoft/xtremedistil-l6-h256-uncased",
        projection_layers: int = 3,
        embed_dims: int = 512,
        max_len: int = 128,
        cls_type: bool = True,
        **kwargs,
    ):
        self.text_model = text_model
        self.projection_layers = projection_layers
        self.embed_dims = embed_dims
        self.max_len = max_len
        self.cls_type = cls_type
        super().__init__(**kwargs)


class TinyCLIPVisionConfig(PretrainedConfig):
    model_type = "vision"

    def __init__(
        self,
        vision_model: str = "edgenext_small",
        projection_layers: int = 3,
        embed_dims: int = 512,
        **kwargs,
    ):
        self.vision_model = vision_model
        self.projection_layers = projection_layers
        self.embed_dims = embed_dims
        super().__init__(**kwargs)


class TinyCLIPConfig(PretrainedConfig):
    model_type = "clip"

    def __init__(
        self,
        text_model: str = "microsoft/xtremedistil-l6-h256-uncased",
        vision_model: str = "edgenext_small",
        projection_layers: int = 3,
        embed_dim: int = 512,
        max_len: int = 128,
        cls_type: bool = True,
        freeze_vision_base: bool = False,
        freeze_text_base: bool = True,
        loss_type: str = "cyclip",
        **kwargs,
    ):
        self.text_config = TinyCLIPTextConfig(
            text_model=text_model,
            projection_layers=projection_layers,
            embed_dims=embed_dim,
            max_len=max_len,
            cls_type=cls_type,
        )
        self.vision_config = TinyCLIPVisionConfig(
            vision_model=vision_model, projection_layers=projection_layers, embed_dims=embed_dim
        )
        self.freeze_vision_base = freeze_vision_base
        self.freeze_text_base = freeze_text_base
        self.loss_type = loss_type
        super().__init__(**kwargs)


class TrainerConfig(pydantic.BaseModel):
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 5e-4
    lr_scheduler: bool = True
    accumulate_grad_batches: int = 1
    temperature: float = 1.0
    vision_freeze_layers: int = 2
    lambda_1: float = 1.0
    lambda_2: float = 1.0

    val_check_interval: int = 1000
    log_every_n_steps: int = 100
    debug: bool = False

    run_openai_clip: bool = False

    _model_config: TinyCLIPConfig = TinyCLIPConfig()
    _data_config: DataConfig = DataConfig()
