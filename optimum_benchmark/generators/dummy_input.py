from logging import getLogger
from typing import Dict, List, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from optimum_benchmark.generators.base import TASKS_TO_GENERATORS, TaskGenerator


LOGGER = getLogger("dummy_dataset")


class DummyInputGenerator:
    task_generator: TaskGenerator

    def __init__(
        self,
        task: str,
        model_shapes: Dict[str, int],
        input_shapes: Dict[str, int],
    ):
        if task in TASKS_TO_GENERATORS:
            LOGGER.info(f"Using {TASKS_TO_GENERATORS[task]} generator")
            self.task_generator = TASKS_TO_GENERATORS[task](input_shapes, model_shapes)
        else:
            raise NotImplementedError(
                f"This task {task} is not implemented. "
                f"Available tasks {list(TASKS_TO_GENERATORS.keys())}. "
                "If you want to add this task, please submit a PR or a feature request."
            )

    def generate(
        self, mode: str
    ) -> Tuple[Dict[str, Union["torch.Tensor", List[str]]], Dict[str, int]]:
        task_input = self.task_generator.generate(mode=mode, with_labels=False)
        static_shapes = self.task_generator.get_static_shapes()

        return task_input, static_shapes
