from src import data
from src import config
from src import vision_model
from src import tokenizer as tk
from src.lightning_module import LightningModule
from src import loss
from src import models


def train(config: config.TrainerConfig):
    transform = vision_model.get_vision_transform(config._model_config.vision_config)
    tokenizer = tk.Tokenizer(config._model_config.text_config)
    train_dl, valid_dl = data.get_dataset(
        transform=transform, tokenizer=tokenizer, hyper_parameters=config  # type: ignore
    )
    vision_encoder = models.TinyCLIPVisionEncoder(config=config._model_config.vision_config)
    text_encoder = models.TinyCLIPTextEncoder(config=config._model_config.text_config)

    lightning_module = LightningModule(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        loss_fn=loss.get_loss(config._model_config.loss_type),
        hyper_parameters=config,
        len_train_dl=len(train_dl),
    )
