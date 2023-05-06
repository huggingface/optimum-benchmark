from dataclasses import dataclass
from logging import getLogger
from typing import Dict


from torch import Tensor
from transformers import AutoFeatureExtractor, AutoConfig
from optimum.utils import DummyAudioInputGenerator, NormalizedConfig

from src.input.base import InputGenerator, InputConfig

INPUT_NAME = 'audio'
LOGGER = getLogger(INPUT_NAME)


@dataclass
class AudioConfig(InputConfig):
    name: str = INPUT_NAME

    batch_size: int = 2
    feature_size: int = 80
    nb_max_frames: int = 3000
    audio_sequence_length: int = 16000


class AudioGenerator(InputGenerator):
    NAME = INPUT_NAME

    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)
        self.normalized_config = NormalizedConfig(
            AutoConfig.from_pretrained(self.model))
        self.input_names = AutoFeatureExtractor.from_pretrained(
            self.model).model_input_names

    def configure(self, config: AudioConfig) -> None:
        self.dummy_audio_generator = DummyAudioInputGenerator(
            task=self.task,
            normalized_config=self.normalized_config,
            batch_size=config.batch_size,
            feature_size=config.feature_size,
            nb_max_frames=config.nb_max_frames,
            audio_sequence_length=config.audio_sequence_length,
        )

    def generate(self) -> Dict[str, Tensor]:
        dummy_input = dict()
        for input_name in self.input_names:
            # for now, since attention_mask is not suupported in this generator
            if input_name == 'attention_mask':
                continue

            dummy_input[input_name] = self.dummy_audio_generator.generate(
                input_name,
                framework='pt'
            ).to(self.device)

        return dummy_input
