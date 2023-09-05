from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from transformers import PretrainedConfig

from ..generators.model_type_generator import SUPPURTED_MODEL_TYPES, ModelTypeGenerator
from ..generators.task_generator import TASKS_TO_GENERATORS, TaskGenerator

LOGGER = getLogger("input_generator")


class InputGenerator:
    model_type_generator: Optional[ModelTypeGenerator]
    task_generator: Optional[TaskGenerator]

    def __init__(
        self,
        task: str,
        input_shapes: Dict[str, int],
        pretrained_config: Optional["PretrainedConfig"] = None,
    ):
        model_type = pretrained_config.model_type if pretrained_config is not None else task

        if model_type in SUPPURTED_MODEL_TYPES:
            self.used_generator = "model_type"
            LOGGER.info(f"Using {model_type} model type generator")
            self.model_type_generator = ModelTypeGenerator(
                task=task,
                model_type=model_type,
                shapes=input_shapes,
                pretrained_config=pretrained_config,
            )
        elif task in TASKS_TO_GENERATORS:
            self.used_generator = "task"
            LOGGER.info(f"Using {task} task generator")
            self.task_generator = TASKS_TO_GENERATORS[task](
                shapes=input_shapes,
                with_labels=False,
            )
        else:
            raise NotImplementedError(
                f"Neither task {task} nor model type {model_type} are supported. "
                f"Available tasks: {list(TASKS_TO_GENERATORS.keys())}. "
                "If you want to add support for this task, "
                "please submit a PR or a feature request to optimum-benchmark. "
                f"Available model types: {SUPPURTED_MODEL_TYPES}. "
                "If you want to add support for this model type, "
                "please submit a PR or a feature request to optimum."
            )

    # TODO: we can drop the torch dependency here by returning a dict of numpy arrays
    # and then converting them to torch tensors in backend.prepare_for_inference
    def generate(self, mode: str) -> Dict[str, Any]:
        if self.used_generator == "model_type":
            dummy_input = self.model_type_generator.generate()
        elif self.used_generator == "task":
            dummy_input = self.task_generator.generate()

        if mode == "generate":
            if "pixel_values" in dummy_input:
                # image input
                dummy_input = {
                    "pixel_values": dummy_input["pixel_values"],
                }
            elif "input_values" in dummy_input:
                # speech input
                dummy_input = {
                    "input_values": dummy_input["input_values"],
                }
            elif "input_features" in dummy_input:
                # waveform input
                dummy_input = {
                    "input_features": dummy_input["input_features"],
                }
            elif "input_ids" in dummy_input:
                # text input
                dummy_input = {
                    "input_ids": dummy_input["input_ids"],
                }

        return dummy_input
