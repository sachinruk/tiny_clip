from PIL import Image
import timm
from timm import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import PreTrainedModel

from src.config import TinyCLIPConfig, TinyCLIPTextConfig, TinyCLIPVisionConfig
from src import loss


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


def projection_layers(d_in: int, d_out: int, num_layers: int) -> nn.Module:
    layers = []
    for _ in range(num_layers - 1):
        layers.extend([Projection(d_in, d_in), nn.GELU()])
    layers += [Projection(d_in, d_out)]
    return nn.Sequential(*layers)


def mean_pooling(
    text_representation: torch.FloatTensor, attention_mask: torch.LongTensor
) -> torch.FloatTensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(text_representation.size()).float()
    return torch.sum(text_representation * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )  # type: ignore


class TinyCLIPTextEncoder(PreTrainedModel):
    config_class = TinyCLIPTextConfig

    def __init__(self, config: TinyCLIPTextConfig):
        super().__init__(config)
        self.base = transformers.AutoModel.from_pretrained(config.text_model)
        self.cls_type = config.cls_type
        self.projection = projection_layers(
            self.base.config.hidden_size, config.embed_dims, config.projection_layers
        )

    def forward(self, x: dict[str, torch.Tensor]):
        out = self.base(**x).last_hidden_state
        if self.cls_type:
            out = out[:, 0]  # get CLS token output
        else:
            out = mean_pooling(out, x["attention_mask"])  # type: ignore

        projected_vec = self.projection(out)
        return F.normalize(projected_vec, dim=-1)


class TinyCLIPVisionEncoder(PreTrainedModel):
    config_class = TinyCLIPVisionConfig

    def __init__(self, config: TinyCLIPVisionConfig):
        super().__init__(config)
        self.base = timm.create_model(config.vision_model, num_classes=0)
        timm_config = data.resolve_data_config({}, model=self.base)
        self.transform = data.transforms_factory.create_transform(**timm_config)
        self.projection = projection_layers(
            self.base.num_features, config.embed_dims, config.projection_layers
        )

    def forward(self, images: list[Image.Image]):
        x: torch.Tensor = torch.stack([self.transform(image) for image in images])  # type: ignore

        projected_vec = self.projection(self.base(x))
        return F.normalize(projected_vec, dim=-1)


class TinyCLIP(PreTrainedModel):
    config_class = TinyCLIPConfig

    def __init__(self, config: TinyCLIPConfig):
        super().__init__(config)
        self.text_encoder = TinyCLIPTextEncoder(config.text_config)
        self.vision_encoder = TinyCLIPVisionEncoder(config.vision_config)

        if config.freeze_text_base:
            self.text_encoder.base.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        if config.freeze_vision_base:
            self.vision_encoder.base.eval()
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        self.loss_fn = loss.get_loss(config.loss_type)

    def forward(
        self,
        text_input: dict[str, torch.Tensor],
        vision_input: list[Image.Image],
        return_loss: bool = False,
    ) -> dict[str, torch.Tensor]:
        text_output = self.text_encoder(text_input)
        vision_output = self.vision_encoder(vision_input)

        out = {"text_output": text_output, "vision_output": vision_output}

        if return_loss:
            out["loss"] = self.loss_fn(vision_output, text_output)

        return out
