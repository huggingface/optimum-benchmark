from logging import getLogger
from typing import Dict

from datasets import Dataset
from transformers import PretrainedConfig

from optimum_benchmark.generators.base import *


LOGGER = getLogger("dummy_dataset")


class DummyDatasetGenerator:
    def __init__(self, dataset_shapes: Dict[str, int]):
        self.dataset_shapes = dataset_shapes

    def generate(
        self,
        task: str,
        pretrained_config: PretrainedConfig,
    ) -> Dataset:
        dataset_config = parse_pretrained_config(pretrained_config)

        if task == "text-classification":
            input_ids = generate_input_ids(
                vocab_size=dataset_config["vocab_size"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            attention_mask = generate_attention_mask(
                input_ids_or_values=input_ids,
            )
            labels = generate_sequence_labels(
                num_labels=dataset_config["num_labels"],
                batch_size=self.dataset_shapes["dataset_size"],
            )

            return Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        elif task == "token-classification":
            input_ids = generate_input_ids(
                vocab_size=dataset_config["vocab_size"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            labels = generate_token_labels(
                num_labels=dataset_config["num_labels"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            return Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )

        elif task == "text-generation":
            input_ids = generate_input_ids(
                vocab_size=dataset_config["vocab_size"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            return Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "labels": input_ids,
                }
            )

        elif task == "question-answering":
            input_ids = generate_input_ids(
                vocab_size=dataset_config["vocab_size"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            attention_mask = generate_attention_mask(
                input_ids_or_values=input_ids,
            )
            start_positions = generate_start_positions(
                batch_size=self.dataset_shapes["dataset_size"],
            )
            end_positions = generate_end_positions(
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            return Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "start_positions": start_positions,
                    "end_positions": end_positions,
                }
            )

        elif task == "fill-mask":
            input_ids = generate_input_ids(
                vocab_size=dataset_config["vocab_size"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            attention_mask = generate_attention_mask(
                input_ids_or_values=input_ids,
            )
            return Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids,
                }
            )

        elif task == "multiple-choice":
            input_ids = generate_input_ids(
                vocab_size=dataset_config["vocab_size"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            attention_mask = generate_attention_mask(
                input_ids_or_values=input_ids,
            )

        else:
            raise NotImplementedError(
                f"Training benchmark not implemented for task {task}."
                "Please submit a PR to add support for this task."
            )
