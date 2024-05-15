from typing import Dict

from datasets import Dataset

from .task_generator import TASKS_TO_GENERATORS, TaskGenerator


class DatasetGenerator:
    task_generator: TaskGenerator

    def __init__(self, task: str, dataset_shapes: Dict[str, int], model_shapes: Dict[str, int]) -> None:
        dataset_shapes["batch_size"] = dataset_shapes["dataset_size"]

        if task in TASKS_TO_GENERATORS:
            shapes = {**dataset_shapes, **model_shapes}
            self.task_generator = TASKS_TO_GENERATORS[task](shapes=shapes, with_labels=True)
        else:
            raise NotImplementedError(
                f"Task {task} is supported. \n"
                f"Available tasks: {list(TASKS_TO_GENERATORS.keys())}. \n"
                "If you want to add support for this task, "
                "please submit a PR or a feature request to optimum-benchmark. \n"
            )

    def __call__(self) -> Dataset:
        task_dataset = self.task_generator()
        task_dataset = Dataset.from_dict(task_dataset)
        task_dataset.set_format(type="torch", columns=list(task_dataset.features.keys()))
        return task_dataset
