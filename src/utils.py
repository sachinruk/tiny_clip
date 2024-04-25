import datetime

import pytorch_lightning as pl
from pytorch_lightning import loggers

from src import config


def _get_wandb_logger(trainer_config: config.TrainerConfig):
    name = f"{config.MODEL_NAME}-{datetime.datetime.now()}"
    if trainer_config.debug:
        name = "debug-" + name
    return loggers.WandbLogger(
        entity=config.WANDB_ENTITY,
        save_dir=config.WANDB_LOG_PATH,
        project=config.MODEL_NAME,
        name=name,
        config=trainer_config._model_config.to_dict(),
    )


def get_trainer(trainer_config: config.TrainerConfig):
    return pl.Trainer(
        max_epochs=trainer_config.epochs if not trainer_config.debug else 1,
        logger=_get_wandb_logger(trainer_config),
        log_every_n_steps=trainer_config.log_every_n_steps,
        gradient_clip_val=1.0,
        limit_train_batches=5 if trainer_config.debug else 1.0,
        limit_val_batches=5 if trainer_config.debug else 1.0,
        accelerator="auto",
    )
