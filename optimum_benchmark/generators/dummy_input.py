from typing import Dict, Tuple
from logging import getLogger
from torch import Tensor
import inspect
import torch


from transformers.onnx.utils import get_preprocessor
from optimum.utils import NormalizedConfigManager
from optimum.exporters import TasksManager


from optimum_benchmark.backends.base import Backend


LOGGER = getLogger("dummy_input")


class DummyInputGenerator:
    def __init__(self, input_shapes: Dict[str, int]):
        self.input_shapes = input_shapes

    def generate(
        self,
        mode: str,
        backend: Backend,
    ) -> Tuple[Dict[str, Tensor], Dict[str, int]]:
        assert mode in ["forward", "generate"], f"mode {mode} not supported"

        if backend.task in ["stable-diffusion", "stable-diffusion-xl"]:
            # patch for stable-diffusion not recognized by TasksManager
            return {
                "prompt": ["This is a sample prompt"] * self.input_shapes["batch_size"]
            }, {"batch_size": self.input_shapes["batch_size"]}

        if backend.task == "text-generation":
            # for LLMs not recognized by TasksManager
            return {
                "input_ids": torch.randint(
                    0,
                    backend.pretrained_config.vocab_size,
                    (
                        self.input_shapes["batch_size"],
                        self.input_shapes["sequence_length"],
                    ),
                    device=backend.device,
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones(
                    (
                        self.input_shapes["batch_size"],
                        self.input_shapes["sequence_length"],
                    ),
                    device=backend.device,
                    dtype=torch.long,
                ),
            }, {
                "batch_size": self.input_shapes["batch_size"],
                "sequence_length": self.input_shapes["sequence_length"],
            }

        # get the normalized config
        normalized_config = NormalizedConfigManager.get_normalized_config_class(
            model_type=backend.pretrained_config.model_type,
        )(backend.pretrained_config)

        # get the onnx config
        onnx_config = TasksManager.get_exporter_config_constructor(
            model_type=backend.pretrained_config.model_type,
            task=backend.task,
            exporter="onnx",
        )(backend.pretrained_config)

        if mode == "forward":
            input_names = list(onnx_config.inputs.keys())
        elif mode == "generate":
            input_names = get_preprocessor(backend.model).model_input_names
        else:
            raise ValueError(f"Unknown mode {mode}")

        generator_classes = onnx_config.DUMMY_INPUT_GENERATOR_CLASSES

        LOGGER.info(f"Generating dummy input for: {input_names}")

        dummy_input = dict()
        dummy_input_shapes = dict()

        for input_name in input_names:
            generator = None

            for generator_class in generator_classes:
                if input_name in generator_class.SUPPORTED_INPUT_NAMES:
                    supported_generator_input_shapes = {
                        input_shape: self.input_shapes[input_shape]
                        for input_shape in self.input_shapes
                        if input_shape
                        in inspect.signature(generator_class.__init__).parameters
                    }
                    generator = generator_class(
                        task=backend.task,
                        normalized_config=normalized_config,
                        **supported_generator_input_shapes,
                    )
                    # we found a generator for this input name, let's use it
                    break

            if generator is None:
                raise ValueError(
                    f"Could not find dummy input generator for {input_name}"
                )

            dummy_input[input_name] = generator.generate(input_name).to(backend.device)
            dummy_input_shapes.update(supported_generator_input_shapes)

            if input_name == "attention_mask":
                # patch for until sparse attention is supported
                dummy_input["attention_mask"] = torch.ones_like(
                    dummy_input["attention_mask"]
                )

        return dummy_input, dummy_input_shapes
