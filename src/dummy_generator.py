import inspect
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

    def generate(self, mode, **input_shapes) -> Dict[str, Tensor]:
        assert mode in ["forward", "generate"], f"mode {mode} not supported"
        assert "batch_size" in input_shapes, "batch_size must be provided in input_shapes"

        # hacky way to get what we need
        auto_config = AutoConfig.from_pretrained(
            self.model, **self.model_kwargs)
        model_type = auto_config.model_type

        # patch for some LLMs model types not recognized by TasksManager
        if model_type in LLM_MODEL_TYPES:
            return {
                "input_ids": torch.ones(
                    (input_shapes["batch_size"], 1), dtype=torch.long, device=self.device
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
                supported_generator_params = inspect.signature(
                    generator_class.__init__
                ).parameters.keys()
                supported_generator_inputs = generator_class.SUPPORTED_INPUT_NAMES

                if input_name in supported_generator_inputs:
                    generator = generator_class(
                        task=self.task,
                        normalized_config=normalized_config,
                        # get the value from kwargs if it exists, otherwise use the default value
                        **{
                            param: input_shapes[param]
                            for param in supported_generator_params
                            if param in input_shapes
                        },
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
