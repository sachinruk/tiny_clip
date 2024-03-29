import io
import multiprocessing as mp
from typing import Optional, Union

import datasets
from PIL import Image
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from src import config


class Tokenizer:
    def __init__(self, model_name: str, max_len: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __call__(self, x: Union[str, list[str]]) -> dict[str, torch.LongTensor]:
        return self.tokenizer(
            x, max_length=self.max_len, truncation=True, padding=True, return_tensors="pt"
        )

    def decode(self, x: dict[str, torch.LongTensor]) -> list[str]:
        return [
            self.tokenizer.decode(sentence[:sentence_len])
            for sentence, sentence_len in zip(x["input_ids"], x["attention_mask"].sum(axis=-1))
        ]


def _get_image_and_caption(item: dict[str, str]) -> Optional[tuple[Image.Image, str]]:
    image_url = item["url"]
    caption = item["caption"]
    try:
        response = requests.get(image_url, timeout=1)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        image = Image.open(io.BytesIO(response.content))
        return image, caption
    except (requests.RequestException, IOError):
        return None


class CollateFn:
    def __init__(self, tokenizer: Tokenizer, transform: transforms.Compose):
        self.tokenizer = tokenizer
        self.transform = transform

    def __call__(
        self, batch: list[Optional[tuple[str, torch.FloatTensor]]]
    ) -> tuple[dict[str, torch.LongTensor], torch.FloatTensor]:
        filtered_batch = [data for data in map(_get_image_and_caption, batch) if data is not None]
        x, y = zip(*filtered_batch)
        tokenized_text = self.tokenizer(list(x))
        return tokenized_text, torch.stack([self.transform(image) for image in y])


def _get_dataloaders(
    train_ds: Dataset,
    valid_ds: Dataset,
    training_config: config.TrainerConfig,
    collate_fn: CollateFn,
) -> tuple[DataLoader, DataLoader]:
    common_params = {
        "batch_size": training_config.batch_size,
        "pin_memory": True,
        "num_workers": mp.cpu_count(),
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
    tokenizer: Tokenizer,
    hyper_parameters: config.TrainerConfig,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    dataset = datasets.load_dataset(
        hyper_parameters.data_config.dataset, split="train", streaming=True
    )
    full_dataset = dataset.shuffle(
        seed=42, buffer_size=hyper_parameters.data_config.buffer_size
    ).take(hyper_parameters.data_config.data_len)
    train_dataset = full_dataset.take(hyper_parameters.data_config.train_len)
    valid_dataset = full_dataset.skip(hyper_parameters.data_config.train_len)

    collate_fn = CollateFn(tokenizer, transform)

    return _get_dataloaders(
        train_ds=train_dataset,
        valid_ds=valid_dataset,
        training_config=hyper_parameters,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
