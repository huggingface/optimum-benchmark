from logging import getLogger
from typing import Dict, List

import torch
from optimum.exporters.tasks import TasksManager
from transformers import PretrainedConfig

LOGGER = getLogger("model_type_generator")

SUPPURTED_MODEL_TYPES: List[str] = list(TasksManager._SUPPORTED_MODEL_TYPE.keys())


class ModelTypeGenerator:
    """A wrapper around optimum's TasksManager to generate dummy inputs
    for a given model type.
    """

    def __init__(
        self,
        task: str,
        model_type: str,
        shapes: Dict[str, int],
        pretrained_config: PretrainedConfig,
    ):
        self.shapes = shapes

        self.onnx_config = TasksManager.get_exporter_config_constructor(
            task=task,
            exporter="onnx",
            model_type=model_type,
        )(pretrained_config)

    def generate(self) -> Dict[str, int]:
        dummy_input = self.onnx_config.generate_dummy_inputs(framework="pt", **self.shapes)

        if "attention_mask" in dummy_input:
            dummy_input["attention_mask"] = torch.ones_like(dummy_input["attention_mask"])

        return dummy_input
