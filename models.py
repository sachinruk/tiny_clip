import dataclasses
import json

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


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
    )


class TextEncoder(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        d_in: int,
        d_out: int,
        n_projection_layers: int,
        cls_token: bool = False,
    ):
        super().__init__()
        self.base = base
        self.cls_token = cls_token
        self.projection = projection_layers(d_in, d_out, n_projection_layers)
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(**x).last_hidden_state
        if self.cls_token:
            out = out[:, 0]  # get CLS token output
        else:
            out = mean_pooling(out, x["attention_mask"])

        projected_vec = self.projection(out)
        return F.normalize(projected_vec, dim=-1)


class VisionEncoder(nn.Module):
    def __init__(self, base: nn.Module, d_in: int, d_out: int, n_projection_layers: int):
        super().__init__()
        self.base = base
        self.projection = projection_layers(d_in, d_out, n_projection_layers)

        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        projected_vec = self.projection(self.base(x))
        return F.normalize(projected_vec, dim=-1)


class Tokenizer:
    def __init__(self, tokenizer, max_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, x: str) -> transformers.AutoTokenizer:
        return self.tokenizer(
            x, max_length=self.max_len, truncation=True, padding=True, return_tensors="pt"
        )

    def decode(self, x: dict[str, torch.LongTensor]) -> list[str]:
        return [
            self.tokenizer.decode(sentence[:sentence_len])
            for sentence, sentence_len in zip(x["input_ids"], x["attention_mask"].sum(axis=-1))
        ]


@dataclasses.dataclass(frozen=True)
class CLIPConfig:
    cls_token: bool = True
    n_projection_layers: int = 3
    embed_dims: int = 512
    vision_model: str = "edgenext_small"
    text_model: str = "microsoft/xtremedistil-l6-h256-uncased"
    max_len: int = 128


def get_model():
    with open("./clip_config.json", "r") as f:
        config = CLIPConfig(**json.load(f))

    # load text model and tokenizer
    text_config = transformers.AutoConfig.from_pretrained("./text_model_config/")
    text_base = transformers.AutoModel.from_config(text_config)
    tokenizer = Tokenizer(
        transformers.AutoTokenizer.from_pretrained("./tokenizer/"), config.max_len
    )
    text_encoder = TextEncoder(
        text_base,
        text_base.config.hidden_size,
        config.embed_dims,
        config.n_projection_layers,
        config.cls_token,
    )
    text_encoder.load_state_dict(torch.load("./text.ckpt", map_location=torch.device("cpu")))

    # load vision model and image transform
    image_base = timm.create_model(config.vision_model, num_classes=0)
    timm_config = timm.data.resolve_data_config({}, model=image_base)
    transform = timm.data.transforms_factory.create_transform(**timm_config)
    vision_encoder = VisionEncoder(
        image_base, image_base.num_features, config.embed_dims, config.n_projection_layers
    )
    vision_encoder.load_state_dict(torch.load("./vision.ckpt", map_location=torch.device("cpu")))

    return text_encoder, tokenizer, vision_encoder, transform
