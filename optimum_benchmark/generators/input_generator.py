from logging import getLogger
from typing import Any, Dict

from .task_generator import TASKS_TO_GENERATORS, TaskGenerator

LOGGER = getLogger("input-generator")


class InputGenerator:
    task_generator: TaskGenerator

    def __init__(self, task: str, input_shapes: Dict[str, int]):
        if task in TASKS_TO_GENERATORS:
            LOGGER.info(f"Using {task} task generator")
            self.task_generator = TASKS_TO_GENERATORS[task](
                shapes=input_shapes,
                with_labels=False,
            )
        else:
            raise NotImplementedError(
                f"Task {task} is not supported. "
                f"Available tasks: {list(TASKS_TO_GENERATORS.keys())}. "
                "If you want to add support for this task, "
                "please submit a PR or a feature request to optimum-benchmark. "
            )

    def generate(self, mode: str) -> Dict[str, Any]:
        task_input = self.task_generator.generate()

        if mode == "generate":
            if "pixel_values" in task_input:
                # image input
                task_input = {
                    "inputs": task_input["pixel_values"],
                }
            elif "input_values" in task_input:
                # speech input
                task_input = {
                    "inputs": task_input["input_values"],
                }
            elif "input_features" in task_input:
                # waveform input
                task_input = {
                    "inputs": task_input["input_features"],
                }
            elif "input_ids" in task_input:
                # text input
                task_input = {
                    "inputs": task_input["input_ids"],
                }

        return task_input
