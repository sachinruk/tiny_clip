from typing import Union

import torch
from transformers import AutoTokenizer

from src.config import TinyCLIPTextConfig


class Tokenizer:
    def __init__(self, text_config: TinyCLIPTextConfig) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(text_config.text_model)
        self.max_len = text_config.max_len

    def __call__(self, x: Union[str, list[str]]) -> dict[str, torch.LongTensor]:
        return self.tokenizer(
            x, max_length=self.max_len, truncation=True, padding=True, return_tensors="pt"
        )  # type: ignore

    def decode(self, x: dict[str, torch.LongTensor]) -> list[str]:
        return [
            self.tokenizer.decode(sentence[:sentence_len])
            for sentence, sentence_len in zip(x["input_ids"], x["attention_mask"].sum(axis=-1))
        ]
