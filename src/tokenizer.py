from typing import Union

import torch
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_name: str, max_len: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __call__(self, x: Union[str, list[str]]) -> dict[str, torch.LongTensor]:
        return self.tokenizer(
            x, max_length=self.max_len, truncation=True, padding=True, return_tensors="pt"
        )  # type: ignore

    def decode(self, x: dict[str, torch.LongTensor]) -> list[str]:
        return [
            self.tokenizer.decode(sentence[:sentence_len])
            for sentence, sentence_len in zip(x["input_ids"], x["attention_mask"].sum(axis=-1))
        ]
