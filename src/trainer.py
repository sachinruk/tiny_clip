from src import data
from src import config
from src import vision_model


def train(config: config.TrainerConfig):
    train_dl, valid_dl = data.get_dataset()
