from logging import getLogger
from typing import Dict, Optional, Union

from datasets import Dataset
from transformers import (
    PretrainedConfig,
    PreTrainedTokenizer,
    ImageProcessingMixin,
    FeatureExtractionMixin,
    ProcessorMixin,
)


from optimum_benchmark.generators.utils import *


LOGGER = getLogger("dummy_dataset")


class DummyDatasetGenerator:
    def __init__(self, dataset_shapes: Dict[str, int]):
        self.dataset_shapes = dataset_shapes

    def generate(
        self,
        task: str,
        pretrained_config: PretrainedConfig,
        pretrained_preprocessor: Optional[
            Union[
                PreTrainedTokenizer,
                ImageProcessingMixin,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ],
    ) -> Dataset:
        model_config = get_model_config(
            pretrained_config=pretrained_config,
            pretrained_preprocessor=pretrained_preprocessor,
        )

        if task == "text-classification":
            input_ids = generate_input_ids(
                vocab_size=model_config["vocab_size"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            attention_mask = generate_attention_mask(
                input_ids_or_values=input_ids,
            )
            labels = generate_sequence_labels(
                num_labels=model_config["num_labels"],
                batch_size=self.dataset_shapes["dataset_size"],
            )
            text_classification_dataset = Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

            return text_classification_dataset

        elif task == "token-classification":
            input_ids = generate_input_ids(
                vocab_size=model_config["vocab_size"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            attention_mask = generate_attention_mask(
                input_ids_or_values=input_ids,
            )
            labels = generate_token_labels(
                num_labels=model_config["num_labels"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            token_classification_dataset = Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

            return token_classification_dataset

        elif task == "text-generation":
            input_ids = generate_input_ids(
                vocab_size=model_config["vocab_size"],
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
                vocab_size=model_config["vocab_size"],
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
            question_answering_dataset = Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "start_positions": start_positions,
                    "end_positions": end_positions,
                }
            )

            return question_answering_dataset

        elif task == "fill-mask":
            input_ids = generate_input_ids(
                vocab_size=model_config["vocab_size"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
            )
            attention_mask = generate_attention_mask(
                input_ids_or_values=input_ids,
            )
            fill_mask_dataset = Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids,
                }
            )

            return fill_mask_dataset

        elif task == "multiple-choice":
            input_ids = generate_multiple_choice_input_ids(
                vocab_size=model_config["vocab_size"],
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
                num_choices=self.dataset_shapes["num_choices"],
            )
            token_type_ids = generate_multiple_choice_token_type_ids(
                batch_size=self.dataset_shapes["dataset_size"],
                sequence_length=self.dataset_shapes["sequence_length"],
                num_choices=self.dataset_shapes["num_choices"],
            )
            attention_mask = generate_attention_mask(
                input_ids_or_values=input_ids,
            )
            label = generate_multiple_choice_labels(
                num_choices=self.dataset_shapes["num_choices"],
                batch_size=self.dataset_shapes["dataset_size"],
            )
            multiple_choice_dataset = Dataset.from_dict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "label": label,
                }
            )

            return multiple_choice_dataset

        elif task == "image-classification":
            pixel_values = (
                generate_pixel_values(
                    batch_size=self.dataset_shapes["dataset_size"],
                    num_channels=model_config["num_channels"],
                    height=model_config["height"],
                    width=model_config["width"],
                )
                / 255
            )
            labels = generate_sequence_labels(
                num_labels=model_config["num_labels"],
                batch_size=self.dataset_shapes["dataset_size"],
            )
            image_classification_dataset = Dataset.from_dict(
                {
                    "pixel_values": pixel_values,
                    "labels": labels,
                }
            )

            return image_classification_dataset

        elif task == "object-detection":
            pixel_values = (
                generate_pixel_values(
                    batch_size=self.dataset_shapes["dataset_size"],
                    num_channels=model_config["num_channels"],
                    height=model_config["height"],
                    width=model_config["width"],
                )
                / 255
            )
            labels = generate_object_detection_labels(
                batch_size=self.dataset_shapes["dataset_size"],
                num_labels=model_config["num_labels"],
                num_queries=model_config["num_queries"],
            )
            object_detection_dataset = Dataset.from_dict(
                {
                    "pixel_values": pixel_values,
                    "labels": labels,
                },
            )
            object_detection_dataset.set_format(
                type="torch",
                columns=["pixel_values", "labels"],
            )

            return object_detection_dataset

        elif task == "semantic-segmentation":
            pixel_values = (
                generate_pixel_values(
                    batch_size=self.dataset_shapes["dataset_size"],
                    num_channels=model_config["num_channels"],
                    height=model_config["height"],
                    width=model_config["width"],
                )
                / 255
            )
            labels = generate_semantic_segmentation_labels(
                batch_size=self.dataset_shapes["dataset_size"],
                height=model_config["height"],
                width=model_config["width"],
                num_labels=model_config["num_labels"],
            )

            semantic_segmentation_dataset = Dataset.from_dict(
                {
                    "pixel_values": pixel_values,
                    "labels": labels,
                }
            )

            return semantic_segmentation_dataset

        else:
            raise NotImplementedError(
                f"Training benchmark not implemented for task {task}."
                "Please submit a PR to add support for this task."
            )
