from src import config
import json


def test_trainer_config():
    trainer_config = config.TrainerConfig.model_validate_json(
        json.dumps({"epochs": 21, "_model_config": {"text_config": {"text_model": "test"}}})
    )

    assert trainer_config.epochs == 21
    assert trainer_config._model_config.text_config.text_model == "test"
    assert hasattr(trainer_config._model_config.text_config, "max_len")
    assert trainer_config._model_config.vision_config == config.TinyCLIPVisionConfig()
