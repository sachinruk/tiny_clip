import torch
from torch import nn
import torch.nn.functional as F


def metrics(similarity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc


def get_similarity_matrix(
    image_features: torch.Tensor, text_features: torch.Tensor
) -> torch.Tensor:
    return image_features @ text_features.T


def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def contrastive_sigmoid_loss(logits):
    return F.binary_cross_entropy_with_logits(logits, torch.eye(len(logits)), reduction="mean")


class CLIPLoss(nn.Module):
    def __init__(self, logit_temperature: float = -1.0):
        super().__init__()
        self.logit_temperature = nn.Parameter(torch.tensor(logit_temperature))

    def forward(self, similarity_matrix: torch.Tensor):
        temperature = self.logit_temperature.sigmoid()

        caption_loss = contrastive_loss(similarity_matrix / temperature, dim=0)
        image_loss = contrastive_loss(similarity_matrix / temperature, dim=1)

        return 0.5 * (caption_loss + image_loss)


class CyCLIPLoss(nn.Module):
    def __init__(self, logit_temperature: float = -1.0):
        super().__init__()
        self.logit_temperature = nn.Parameter(torch.tensor(logit_temperature))
        self.lambda_1: float = 1.0
        self.lambda_2: float = 1.0

    def forward(
        self,
        similarity_matrix: torch.Tensor,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        temperature = self.logit_temperature.sigmoid()
        caption_loss = contrastive_loss(similarity_matrix / temperature, dim=0)
        image_loss = contrastive_loss(similarity_matrix / temperature, dim=1)

        symmetry_loss = F.mse_loss(similarity_matrix, similarity_matrix.T)
        modality_difference_loss = F.mse_loss(
            image_features @ image_features.T, text_features @ text_features.T
        )

        return (
            0.5 * (caption_loss + image_loss)
            + self.lambda_1 * symmetry_loss
            + self.lambda_2 * modality_difference_loss
        )


class SigLIPLoss(nn.Module):
    def __init__(self, logit_temperature: float = -1.0):
        super().__init__()
        self.logit_temperature = nn.Parameter(torch.tensor(logit_temperature))

    def forward(self, similarity_matrix: torch.Tensor):
        temperature = self.logit_temperature.sigmoid()
        return contrastive_sigmoid_loss(similarity_matrix / temperature)


class CySigLIPLoss(nn.Module):
    def __init__(self, logit_temperature: float = -1.0):
        super().__init__()
        self.logit_temperature = nn.Parameter(torch.tensor(logit_temperature))
        self.lambda_1: float = 1.0
        self.lambda_2: float = 1.0

    def forward(
        self,
        similarity_matrix: torch.Tensor,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        temperature = self.logit_temperature.sigmoid()
        loss = contrastive_sigmoid_loss(similarity_matrix / temperature)

        symmetry_loss = F.mse_loss(similarity_matrix, similarity_matrix.T)
        modality_difference_loss = F.mse_loss(
            image_features @ image_features.T, text_features @ text_features.T
        )

        return loss + self.lambda_1 * symmetry_loss + self.lambda_2 * modality_difference_loss


def get_loss(loss_type: str):
    loss_functions = {
        "clip": CLIPLoss(),
        "cyclip": CyCLIPLoss(),
        "sigmoid": SigLIPLoss(),
        "cyclic_sigmoid": CySigLIPLoss(),
    }
    if loss_type in loss_functions:
        return loss_functions[loss_type]
    else:
        raise ValueError("Invalid loss type")
