from logging import getLogger
from typing import Dict

import torch
from torch import Tensor
from transformers import AutoConfig
from optimum.exporters import TasksManager
from transformers.onnx.utils import get_preprocessor


LOGGER = getLogger("dummy_input_generator")


class DummyInputGenerator:
    def __init__(self, model: str, task: str, device: str) -> None:
        self.model = model
        self.task = task
        self.device = device

    def generate(self, mode) -> Dict[str, Tensor]:
        # LOGGER.info(f"Generating dummy input")

        auto_config = AutoConfig.from_pretrained(self.model)
        model_type = auto_config.model_type
        onnx_config = TasksManager._SUPPORTED_MODEL_TYPE[model_type]["onnx"][self.task](
            auto_config
        )
        normalized_config = onnx_config.NORMALIZED_CONFIG_CLASS(auto_config)  # type: ignore
        # LOGGER.info(f"\t+ Using {onnx_config.__class__.__name__} as onnx config")

        if mode == "forward":
            input_names = list(onnx_config.inputs.keys())  # type: ignore
        elif mode == "generate":
            input_names = get_preprocessor(self.model).model_input_names  # type: ignore
        else:
            raise ValueError(f"Unknown mode {mode}")

        # LOGGER.info(f"\t+ Using {input_names} as model input names")

        dummy_input = dict()
        for input_name in input_names:
            dummy_input_generator = None

            for dummy_input_generator_class in onnx_config.DUMMY_INPUT_GENERATOR_CLASSES:  # type: ignore
                if input_name in dummy_input_generator_class.SUPPORTED_INPUT_NAMES:  # type: ignore
                    dummy_input_generator = dummy_input_generator_class(
                        task=self.task,
                        normalized_config=normalized_config,
                    )

            if dummy_input_generator is None:
                raise ValueError(
                    f"Could not find dummy input generator for {input_name}"
                )

            # LOGGER.info(
            #     f"\t+ Generating dummy input for {input_name} using {dummy_input_generator.__class__.__name__}"
            # )

            dummy_input[input_name] = dummy_input_generator.generate(
                input_name, framework="pt"
            ).to(self.device)

            # this is for bettertransformer since it does not support random attention mask
            if input_name == "attention_mask":
                dummy_input["attention_mask"] = torch.ones_like(
                    dummy_input["input_ids"]
                )

        return dummy_input
