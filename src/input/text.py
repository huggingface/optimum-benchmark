from src.input.base import InputConfig
from dataclasses import dataclass
from logging import getLogger
from typing import Dict

from torch import Tensor
from transformers import AutoTokenizer, AutoConfig
from optimum.utils import NormalizedTextConfig, DummyTextInputGenerator

from src.input.base import InputGenerator, InputConfig


INPUT_NAME = 'text'
LOGGER = getLogger(INPUT_NAME)


@dataclass
class TextConfig(InputConfig):
    name: str = INPUT_NAME

    batch_size: int = 8
    sequence_length: int = 96
    num_choices: int = 4
    sparsity: float = 0.0


class TextGenerator(InputGenerator):
    NAME: str = INPUT_NAME

    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

    def configure(self, config: TextConfig) -> None:
        normalized_config = NormalizedTextConfig(
            AutoConfig.from_pretrained(self.model))

        self.input_names = AutoTokenizer.from_pretrained(
            self.model).model_input_names

        self.dummy_text_generator = DummyTextInputGenerator(
            task=self.task,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            # num_choices=config.num_choices,
            normalized_config=normalized_config,
        )

    def generate(self) -> Dict[str, Tensor]:
        dummy_input = dict()
        for input_name in self.input_names:
            dummy_input[input_name] = self.dummy_text_generator.generate(
                input_name,
                framework='pt'
            )
        return dummy_input

        # if config.sparsity > 0:
        #     # apply sparse mask
        #     mask = torch.rand(
        #         (config.batch_size, config.sequence_length))
        #     attention_mask[mask < config.sparsity] = 0
        #     # force right padding
        #     attention_mask, _ = attention_mask.sort(dim=-1, descending=True)
