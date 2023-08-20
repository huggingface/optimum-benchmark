from typing import Dict, List
from logging import getLogger

from transformers import PretrainedConfig
from optimum.exporters.tasks import TasksManager


LOGGER = getLogger("model_type_generator")

SUPPURTED_MODEL_TYPES: List[str] = list(TasksManager._SUPPORTED_MODEL_TYPE.keys())


class ModelTypeGenerator:
    """
    A wrapper around optimum's TasksManager to generate dummy inputs for a given model type.
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

    @staticmethod
    def check_model_type_support(model_type: str) -> bool:
        return model_type in SUPPURTED_MODEL_TYPES

    def generate(self) -> Dict[str, int]:
        return self.onnx_config.generate_dummy_inputs(framework="pt", **self.shapes)


if __name__ == "__main__":
    from transformers import AutoConfig

    pretrained_config = AutoConfig.from_pretrained("gpt2")

    assert ModelTypeGenerator.check_model_type_support("gpt2")

    model_input_generator = ModelTypeGenerator(
        task="text-generation",
        model_type="gpt2",
        shapes={
            "batch_size": 1,
            "sequence_length": 100,
        },
        pretrained_config=pretrained_config,
    )

    print(model_input_generator.generate())
