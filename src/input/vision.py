from dataclasses import dataclass
from logging import getLogger
from typing import Dict


from torch import Tensor
from transformers import AutoConfig, AutoImageProcessor
from optimum.utils import NormalizedVisionConfig, DummyVisionInputGenerator

from src.input.base import InputGenerator, InputConfig

INPUT_NAME = 'vision'
LOGGER = getLogger(INPUT_NAME)


@dataclass
class AudioConfig(InputConfig):
    name: str = INPUT_NAME

    batch_size: int = 2
    # only relevant when can't be found in the normalized config
    num_channels: int = 3
    width: int = 224
    height: int = 224


class VisionGenerator(InputGenerator):
    NAME = INPUT_NAME

    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)
        self.normalized_config = NormalizedVisionConfig(
            AutoConfig.from_pretrained(self.model))
        self.input_names = AutoImageProcessor.from_pretrained(
            self.model).model_input_names

    def configure(self, config: AudioConfig) -> None:
        self.dummy_audio_generator = DummyVisionInputGenerator(
            task=self.task,
            normalized_config=self.normalized_config,
            batch_size=config.batch_size,
            num_channels=config.num_channels,
            width=config.width,
            height=config.height,
        )

    def generate(self) -> Dict[str, Tensor]:
        dummy_input = dict()
        for input_name in self.input_names:
            # pixel_mask is still unsupported in ORTModelForxxx
            if input_name == 'pixel_mask':
                continue

            dummy_input[input_name] = self.dummy_audio_generator.generate(
                input_name,
                framework='pt'
            ).to(self.device)

        return dummy_input
