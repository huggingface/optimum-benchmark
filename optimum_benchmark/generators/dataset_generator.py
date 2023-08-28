from logging import getLogger
from typing import Dict

from datasets import Dataset

from optimum_benchmark.generators.task_generator import (
    TASKS_TO_GENERATORS,
    TaskGenerator,
)

LOGGER = getLogger("dataset_generator")


class DatasetGenerator:
    task_generator: TaskGenerator

    def __init__(self, task: str, dataset_shapes: Dict[str, int]):
        dataset_shapes["batch_size"] = dataset_shapes.pop("dataset_size")

        if task in TASKS_TO_GENERATORS:
            LOGGER.info(f"Using {task} task generator")
            self.task_generator = TASKS_TO_GENERATORS[task](
                shapes=dataset_shapes,
                with_labels=True,
            )
        else:
            raise NotImplementedError(
                f"Task {task} is supported. \n"
                f"Available tasks: {list(TASKS_TO_GENERATORS.keys())}. \n"
                "If you want to add support for this task, "
                "please submit a PR or a feature request to optimum-benchmark. \n"
            )

    def generate(self) -> Dataset:
        task_dataset = self.task_generator.generate()
        task_dataset = Dataset.from_dict(task_dataset)
        task_dataset.set_format(
            type="torch",  # for now we're using pytorch tensors
            columns=list(task_dataset.features.keys()),
        )

        return task_dataset
