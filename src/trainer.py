import os

from src import config
from src import data
from src import loss
from src import models
from src import tokenizer as tk
from src import vision_model
from src import utils
from src.lightning_module import LightningModule


def _upload_model_to_hub(
    vision_encoder: models.TinyCLIPVisionEncoder, text_encoder: models.TinyCLIPTextEncoder
):
    vision_encoder.save_pretrained(
        str(config.MODEL_PATH),
        variant="vision_encoder",
        safe_serialization=True,
        push_to_hub=True,
        repo_id="debug-clip-model",
    )
    text_encoder.save_pretrained(
        str(config.MODEL_PATH),
        variant="text_encoder",
        safe_serialization=True,
        push_to_hub=True,
        repo_id="debug-clip-model",
    )


def train(trainer_config: config.TrainerConfig):
    if "HF_TOKEN" not in os.environ:
        raise ValueError("Please set the HF_TOKEN environment variable.")
    transform = vision_model.get_vision_transform(trainer_config._model_config.vision_config)
    tokenizer = tk.Tokenizer(trainer_config._model_config.text_config)
    train_dl, valid_dl = data.get_dataset(
        transform=transform, tokenizer=tokenizer, hyper_parameters=trainer_config  # type: ignore
    )
    vision_encoder = models.TinyCLIPVisionEncoder(config=trainer_config._model_config.vision_config)
    text_encoder = models.TinyCLIPTextEncoder(config=trainer_config._model_config.text_config)

    lightning_module = LightningModule(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        loss_fn=loss.get_loss(trainer_config._model_config.loss_type),
        hyper_parameters=trainer_config,
        len_train_dl=len(train_dl),
    )

    trainer = utils.get_trainer(trainer_config)
    trainer.fit(lightning_module, train_dl, valid_dl)

    _upload_model_to_hub(vision_encoder, text_encoder)


if __name__ == "__main__":
    trainer_config = config.TrainerConfig(debug=True)
    train(trainer_config)
