from typing import Dict, Optional

from datasets import Dataset

from .base import BaseGenerator
from .model_generator import MODEL_TYPE_TO_GENERATORS
from .task_generator import TASKS_TO_GENERATORS


class DatasetGenerator:
    generator: BaseGenerator

    def __init__(
        self,
        task: str,
        dataset_shapes: Dict[str, int],
        model_shapes: Dict[str, int],
        model_type: Optional[str] = None,
    ) -> None:
        # dataset_shapes take precedence over model_shapes
        all_shapes = {**model_shapes, **dataset_shapes}
        all_shapes["batch_size"] = all_shapes.pop("dataset_size", None)

        if model_type in MODEL_TYPE_TO_GENERATORS:
            self.generator = MODEL_TYPE_TO_GENERATORS[model_type](shapes=all_shapes, with_labels=True)
        elif task in TASKS_TO_GENERATORS:
            self.generator = TASKS_TO_GENERATORS[task](shapes=all_shapes, with_labels=True)
        else:
            raise NotImplementedError(
                f"Task {task} is not supported for dataset generation. "
                f"Available tasks: {list(TASKS_TO_GENERATORS.keys())}. "
                f"Available model types: {list(MODEL_TYPE_TO_GENERATORS.keys())}. "
                "If you want to add support for this task or model type, "
                "please submit a PR or a feature request to optimum-benchmark."
            )

    def __call__(self) -> Dataset:
        task_dataset = self.generator()
        task_dataset = Dataset.from_dict(task_dataset)
        task_dataset.set_format(type="torch", columns=list(task_dataset.features.keys()))
        return task_dataset
