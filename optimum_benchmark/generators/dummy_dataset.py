from logging import getLogger
from typing import Dict, Optional

from datasets import Dataset

from optimum_benchmark.generators.base import TASKS_TO_GENERATORS, TaskGenerator


LOGGER = getLogger("dummy_dataset")


class DummyDatasetGenerator:
    task_generator: TaskGenerator

    def __init__(
        self,
        task: str,
        dataset_shapes: Dict[str, int],
        model_shapes: Dict[str, Optional[int]],
    ):
        if task in TASKS_TO_GENERATORS:
            LOGGER.info(f"Using {TASKS_TO_GENERATORS[task]} generator")
            dataset_shapes["batch_size"] = dataset_shapes["dataset_size"]
            self.task_generator = TASKS_TO_GENERATORS[task](dataset_shapes, model_shapes)
        else:
            raise NotImplementedError(
                f"This task {task} is not implemented. "
                f"Available tasks {list(TASKS_TO_GENERATORS.keys())}. "
                "If you want to add this task, please submit a PR or a feature request."
            )

    def generate(self) -> Dataset:
        task_dataset = self.task_generator.generate(
            mode="forward",
            with_labels=True,
        )

        task_dataset = Dataset.from_dict(task_dataset)

        task_dataset.set_format(
            type="torch",
            columns=list(task_dataset.features.keys()),
        )

        return task_dataset
