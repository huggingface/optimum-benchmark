from logging import getLogger
from typing import Dict

import torch
from torch import Tensor
from transformers import AutoConfig
from optimum.exporters import TasksManager


LOGGER = getLogger("dummy_input_generator")


class DummyInputGenerator:
    def __init__(self, model: str, task: str, device: str) -> None:
        self.model = model
        self.task = task
        self.device = device

        self.auto_config = AutoConfig.from_pretrained(self.model)
        model_type = self.auto_config.model_type
        LOGGER.info(f"\t+ Using {model_type} as model type")

        self.onnx_config = TasksManager._SUPPORTED_MODEL_TYPE[model_type]["onnx"][
            self.task
        ](self.auto_config)
        LOGGER.info(f"\t+ Using {self.onnx_config.__class__.__name__} as onnx config")

        self.input_names = list(self.onnx_config.inputs.keys())  # type: ignore
        LOGGER.info(f"\t+ Using {self.input_names} as model input names")

    def generate(self) -> Dict[str, Tensor]:
        LOGGER.info(f"Generating dummy inputs")

        dummy_inputs = dict()
        for input_name in self.input_names:
            dummy_input_generator = None

            for dummy_input_generator_class in self.onnx_config.DUMMY_INPUT_GENERATOR_CLASSES:  # type: ignore
                if input_name in dummy_input_generator_class.SUPPORTED_INPUT_NAMES:  # type: ignore
                    dummy_input_generator = dummy_input_generator_class(
                        task=self.task,
                        normalized_config=self.onnx_config.NORMALIZED_CONFIG_CLASS(  # type: ignore
                            self.auto_config
                        ),
                    )

            if dummy_input_generator is None:
                raise ValueError(
                    f"Could not find dummy input generator for {input_name}"
                )

            LOGGER.info(
                f"\t+ Generating dummy input for {input_name} using {dummy_input_generator.__class__.__name__}"
            )

            dummy_inputs[input_name] = dummy_input_generator.generate(
                input_name, framework="pt"
            ).to(self.device)

            # this is for bettertransformer since it does not support random attention mask
            if input_name == "attention_mask":
                dummy_inputs["attention_mask"] = torch.ones_like(
                    dummy_inputs["input_ids"]
                )

        return dummy_inputs
