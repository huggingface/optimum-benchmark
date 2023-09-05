from logging import getLogger
from typing import Dict, List

import torch
from optimum.exporters.tasks import TasksManager
from transformers import PretrainedConfig

from ..import_utils import is_onnx_available

LOGGER = getLogger("model_type_generator")

EXPORTER = "onnx"  # used for its configs as input generators
SUPPURTED_MODEL_TYPES: List[str] = (
    list(TasksManager._SUPPORTED_MODEL_TYPE.keys()) if is_onnx_available() else []
)  # should be empty if onnx is not available


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
            exporter=EXPORTER,
            model_type=model_type,
        )(pretrained_config)

    def generate(self) -> Dict[str, int]:
        dummy_input = self.onnx_config.generate_dummy_inputs(framework="pt", **self.shapes)

        if "attention_mask" in dummy_input:
            dummy_input["attention_mask"] = torch.ones_like(dummy_input["attention_mask"])

        return dummy_input
