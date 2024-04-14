import torch


def metrics(similarity_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(len(similarity_matrix)).to(similarity_matrix.device)
    img2cap_match_idx = similarity_matrix.argmax(dim=1)
    cap2img_match_idx = similarity_matrix.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc
