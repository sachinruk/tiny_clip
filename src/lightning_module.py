import pytorch_lightning as pl
import torch
import torch.nn as nn

from src import config
from src import loss as loss_utils
from src import metrics
from src import models


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        vision_encoder: models.TinyCLIPVisionEncoder,
        text_encoder: models.TinyCLIPTextEncoder,
        loss_fn: nn.Module,
        hyper_parameters: config.TrainerConfig,
        len_train_dl: int,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.loss_fn = loss_fn
        self.hyper_parameters = hyper_parameters
        self.len_train_dl = len_train_dl

    def common_step(self, batch: dict[str, torch.Tensor], step_kind: str) -> torch.Tensor:
        image_features = self.vision_encoder(batch["images"])
        text_features = self.text_encoder(
            {key: value for key, value in batch.items() if key != "images"}
        )
        similarity_matrix = loss_utils.get_similarity_matrix(image_features, text_features)

        loss = self.loss_fn(similarity_matrix, image_features, text_features)

        img_acc, cap_acc = metrics.metrics(similarity_matrix)

        self.log(f"{step_kind}_loss", loss, on_step=False, on_epoch=True)
        self.log(f"{step_kind}_img_acc", img_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{step_kind}_cap_acc", cap_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, list[str]], *args: list) -> torch.Tensor:
        loss = self.common_step(batch, step_kind="training")
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, list[str]], *args: list):
        _ = self.common_step(batch, step_kind="training")

    def configure_optimizers(self):
        vision_params = [
            {
                "params": self.vision_encoder.projection.parameters(),
                "lr": self.hyper_parameters.learning_rate,
            },
        ]
        caption_params = [
            {
                "params": self.text_encoder.projection.parameters(),
                "lr": self.hyper_parameters.learning_rate,
            },
        ]
        loss_params = [
            {
                "params": self.loss_fn.parameters(),
                "lr": self.hyper_parameters.learning_rate,
            },
        ]

        if not self.hyper_parameters._model_config.freeze_text_base:
            caption_params += [
                {
                    "params": self.text_encoder.base.parameters(),
                    "lr": self.hyper_parameters.learning_rate / 2,
                },
            ]

        if not self.hyper_parameters._model_config.freeze_vision_base:
            vision_params += [
                {
                    "params": self.vision_encoder.base.parameters(),
                    "lr": self.hyper_parameters.learning_rate / 2,
                },
            ]

        optimizer = torch.optim.Adam(
            vision_params + caption_params + loss_params, lr=self.hyper_parameters.learning_rate
        )

        if self.hyper_parameters.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hyper_parameters.learning_rate,
                total_steps=int(self.trainer.estimated_stepping_batches),
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def on_epoch_end(self):
        if self.current_epoch == 0:
            for p in self.vision_encoder.base.parameters():
                p.requires_grad = True
            self.vision_encoder.base.train()
