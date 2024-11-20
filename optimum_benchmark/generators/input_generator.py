from typing import Any, Dict, Optional

from .base import BaseGenerator
from .model_generator import MODEL_TYPE_TO_GENERATORS
from .task_generator import TASKS_TO_GENERATORS


class InputGenerator:
    generator: BaseGenerator

    def __init__(
        self,
        task: str,
        input_shapes: Dict[str, int],
        model_shapes: Dict[str, int],
        model_type: Optional[str] = None,
    ) -> None:
        # input_shapes take precedence over model_shapes
        all_shapes = {**model_shapes, **input_shapes}

        if model_type in MODEL_TYPE_TO_GENERATORS:
            self.generator = MODEL_TYPE_TO_GENERATORS[model_type](shapes=all_shapes, with_labels=False)
        elif task in TASKS_TO_GENERATORS:
            self.generator = TASKS_TO_GENERATORS[task](shapes=all_shapes, with_labels=False)
        else:
            raise NotImplementedError(
                f"Task {task} is not supported for input generation. "
                f"Available tasks: {list(TASKS_TO_GENERATORS.keys())}. "
                f"Available model types: {list(MODEL_TYPE_TO_GENERATORS.keys())}. "
                "If you want to add support for this task or model type, "
                "please submit a PR or a feature request to optimum-benchmark."
            )

    def __call__(self) -> Dict[str, Any]:
        task_input = self.generator()
        return task_input
