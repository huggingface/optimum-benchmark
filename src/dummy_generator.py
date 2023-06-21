from typing import Dict

import torch
from torch import Tensor
from transformers import AutoConfig
from optimum.exporters import TasksManager
from transformers.onnx.utils import get_preprocessor

from src.utils import LLM_MODEL_TYPES


class DummyGenerator:
    def __init__(self, model, task, device, model_kwargs) -> None:
        self.model = model
        self.task = task
        self.device = device
        self.model_kwargs = model_kwargs

    def generate(self, mode, **kwargs) -> Dict[str, Tensor]:
        assert mode in ["forward", "generate"], f"mode {mode} not supported"
        assert "batch_size" in kwargs, "batch_size must be provided in kwargs"

        # hacky way to get what we need
        auto_config = AutoConfig.from_pretrained(
            self.model, **self.model_kwargs)
        model_type = auto_config.model_type

        # patch for some LLMs
        if model_type in LLM_MODEL_TYPES:
            return {
                "input_ids": torch.ones(
                    (kwargs["batch_size"], 1), dtype=torch.long, device=self.device
                )
            }

        onnx_config = TasksManager._SUPPORTED_MODEL_TYPE[model_type]["onnx"][self.task](
            auto_config
        )
        normalized_config = onnx_config.NORMALIZED_CONFIG_CLASS(auto_config)
        generator_classes = onnx_config.DUMMY_INPUT_GENERATOR_CLASSES

        if mode == "forward":
            input_names = list(onnx_config.inputs.keys())
        elif mode == "generate":
            input_names = get_preprocessor(self.model).model_input_names
        else:
            raise ValueError(f"Unknown mode {mode}")

        dummy_inputs = dict()
        for input_name in input_names:
            generator = None
            for generator_class in generator_classes:
                if input_name in generator_class.SUPPORTED_INPUT_NAMES:
                    generator = generator_class(
                        task=self.task,
                        normalized_config=normalized_config,
                        batch_size=kwargs["batch_size"],
                    )

            if generator is None:
                raise ValueError(
                    f"Could not find dummy input generator for {input_name}"
                )

            dummy_inputs[input_name] = generator.generate(
                input_name).to(self.device)

            if input_name == "attention_mask":
                # patch for bettertransformer
                dummy_inputs["attention_mask"] = torch.ones_like(
                    dummy_inputs["input_ids"]
                )

        return dummy_inputs
