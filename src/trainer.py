import os

from huggingface_hub import HfApi
from loguru import logger

from src import config
from src import data
from src import loss
from src import models
from src import tokenizer as tk
from src import vision_model
from src import utils
from src.lightning_module import LightningModule


def _upload_model_to_hub(
    vision_encoder: models.TinyCLIPVisionEncoder,
    text_encoder: models.TinyCLIPTextEncoder,
    debug: bool = False,
):
    vision_encoder.save_pretrained(
        str(config.VISION_MODEL_PATH),
        safe_serialization=True,
    )
    text_encoder.save_pretrained(
        str(config.TEXT_MODEL_PATH),
        safe_serialization=True,
    )

    api = HfApi()
    if debug:
        repo_components = config.REPO_ID.split("/", maxsplit=1)
        repo_components[1] = f"debug-{repo_components[1]}"
        repo_id = "/".join(repo_components)
    else:
        repo_id = config.REPO_ID
    common_hf_api_params = {
        "repo_id": repo_id,
        "repo_type": "model",
    }
    if not api.repo_exists(**common_hf_api_params):
        logger.info(f"Creating repo {repo_id} on Hugging Face Hub.")
        api.create_repo(**common_hf_api_params)  # type: ignore
    logger.info(f"Uploading models in {str(config.MODEL_PATH)} to {repo_id}.")
    api.upload_folder(
        folder_path=config.MODEL_PATH,
        **common_hf_api_params,  # type: ignore
    )  # type: ignore


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

    _upload_model_to_hub(vision_encoder, text_encoder, trainer_config.debug)


if __name__ == "__main__":
    trainer_config = config.TrainerConfig(debug=True)
    train(trainer_config)
