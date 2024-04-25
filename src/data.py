import multiprocessing as mp
import pathlib
from typing import Any

import datasets
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from src import config
from src import tokenizer as tk


class CaptionDatset(Dataset):
    def __init__(self, dataset: datasets.Dataset, img_path: pathlib.Path) -> None:
        self.dataset = dataset
        self.img_path = img_path

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(self.img_path / item["url"].rsplit("/", 1)[-1]).convert("RGB")
        return {"image": image, "caption": item["short_caption"]}


class CollateFn:
    def __init__(self, tokenizer: tk.Tokenizer, transform: transforms.Compose):
        self.tokenizer = tokenizer
        self.transform = transform

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        stacked_images = torch.stack([self.transform(item["image"]) for item in batch])
        tokenized_text = self.tokenizer([item["caption"] for item in batch])

        return {
            "images": stacked_images,
            **tokenized_text,
        }


def _get_dataloaders(
    train_ds: Dataset,
    valid_ds: Dataset,
    training_config: config.TrainerConfig,
    collate_fn: CollateFn,
) -> tuple[DataLoader, DataLoader]:
    common_params = {
        "batch_size": training_config.batch_size,
        "pin_memory": True,
        "num_workers": mp.cpu_count() // 3,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **common_params,
    )
    valid_loader = DataLoader(
        valid_ds,
        shuffle=False,
        drop_last=False,
        **common_params,
    )
    return train_loader, valid_loader


def get_dataset(
    transform: transforms.Compose,
    tokenizer: tk.Tokenizer,
    hyper_parameters: config.TrainerConfig,
) -> tuple[DataLoader, DataLoader]:
    dataset: datasets.Dataset = datasets.load_dataset(
        hyper_parameters._data_config.dataset, split="train"
    )  # type: ignore
    train_test_dataset = dataset.train_test_split(seed=42, test_size=0.1)
    train_ds = CaptionDatset(train_test_dataset["train"], config.IMAGE_DOWNLOAD_PATH)
    valid_ds = CaptionDatset(train_test_dataset["test"], config.IMAGE_DOWNLOAD_PATH)
    collate_fn = CollateFn(tokenizer, transform)

    return _get_dataloaders(
        train_ds=train_ds,
        valid_ds=valid_ds,
        training_config=hyper_parameters,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    # do not want to do these imports in general
    import os

    from tqdm.auto import tqdm

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hyper_parameters = config.TrainerConfig()
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    tokenizer = tk.Tokenizer(
        hyper_parameters._model_config.text_model, hyper_parameters._model_config.max_len
    )
    train_dl, valid_dl = get_dataset(transform, tokenizer, hyper_parameters)

    batch = next(iter(train_dl))
    print({k: v.shape for k, v in batch.items()})  # torch.Size([1, 3, 128, 128])

    for batch in tqdm(train_dl):
        continue
