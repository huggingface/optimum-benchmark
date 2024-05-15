from typing import Any, Dict

from .task_generator import TASKS_TO_GENERATORS, TaskGenerator


class InputGenerator:
    task_generator: TaskGenerator

    def __init__(self, task: str, input_shapes: Dict[str, int], model_shapes: Dict[str, int]) -> None:
        if task in TASKS_TO_GENERATORS:
            shapes = {**input_shapes, **model_shapes}
            self.task_generator = TASKS_TO_GENERATORS[task](shapes=shapes, with_labels=False)
        else:
            raise NotImplementedError(
                f"Task {task} is not supported. "
                f"Available tasks: {list(TASKS_TO_GENERATORS.keys())}. "
                "If you want to add support for this task, "
                "please submit a PR or a feature request to optimum-benchmark. "
            )

    def __call__(self) -> Dict[str, Any]:
        task_input = self.task_generator()
        return task_input
