from functools import partial
from io import BytesIO
import pathlib
from typing import Any

import datasets
from loguru import logger
from PIL import Image
import requests
from tqdm.auto import tqdm

from src import config


def _save_resized_image(example: dict[str, Any], size: tuple[int, int], path: pathlib.Path):
    # Download the image
    image_url = example["url"]
    image_path = path / image_url.rsplit("/", 1)[-1]
    if image_path.exists():
        return

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    # Resize the image
    image_resized = image.resize(size)
    image_resized.save(image_path)


def _get_images(dataset: datasets.Dataset, path: pathlib.Path):
    save_resized_image = partial(_save_resized_image, path=path, size=(256, 256))
    dataset.map(save_resized_image, num_proc=128)


def _check_corrupt_images(image_file: pathlib.Path):
    try:
        with Image.open(image_file) as img:
            img.verify()  # Verify the integrity of the image
    except (IOError, SyntaxError) as e:
        logger.error(f"Corrupt image: {image_file}")


if __name__ == "__main__":
    hyper_parameters = config.TrainerConfig()

    dataset = datasets.load_dataset(
        hyper_parameters._data_config.dataset,
        split="train",
    )

    config.IMAGE_DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    _get_images(dataset, config.IMAGE_DOWNLOAD_PATH)  # type: ignore

    for image in tqdm(config.IMAGE_DOWNLOAD_PATH.iterdir()):
        _check_corrupt_images(image)
