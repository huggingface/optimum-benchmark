from abc import ABC
from logging import getLogger
from typing import Tuple

import torch

LOGGER = getLogger("task_generator")


class TaskGenerator(ABC):
    def __init__(self, shapes, with_labels: bool):
        self.shapes = shapes
        self.with_labels = with_labels

    @staticmethod
    def generate_random_integers(min_value: int, max_value: int, shape: Tuple[int]):
        return torch.randint(min_value, max_value, shape)

    @staticmethod
    def generate_random_floats(min_value: float, max_value: float, shape: Tuple[int]):
        return torch.rand(shape) * (max_value - min_value) + min_value

    def generate(self):
        raise NotImplementedError("Generator must implement generate method")


class TextGenerator(TaskGenerator):
    def input_ids(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["vocab_size"],
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"],
            ),
        )

    def token_type_ids(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["type_vocab_size"],
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"],
            ),
        )

    def attention_mask(self):
        return self.generate_random_integers(
            min_value=1,
            max_value=2,
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"],
            ),
        )


class ImageGenerator(TaskGenerator):
    def pixel_values(self):
        return self.generate_random_floats(
            min_value=0,
            max_value=1,
            shape=(
                self.shapes["batch_size"],
                self.shapes["num_channels"],
                self.shapes["height"],
                self.shapes["width"],
            ),
        )


class AudioGenerator(TaskGenerator):
    def input_values(self):
        return self.generate_random_floats(
            min_value=-1,
            max_value=1,
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"],
            ),
        )

    def input_features(self):
        return self.generate_random_floats(
            min_value=-1,
            max_value=1,
            shape=(
                self.shapes["batch_size"],
                self.shapes["feature_size"],
                self.shapes["nb_max_frames"],
            ),
        )


class TextClassificationGenerator(TextGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"],
            shape=(self.shapes["batch_size"],),
        )

    def generate(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()
        dummy["token_type_ids"] = self.token_type_ids()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class TokenClassificationGenerator(TextGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"],
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"],
            ),
        )

    def generate(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class TextGenerationGenerator(TextGenerator):
    def generate(self):
        dummy = {}
        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.with_labels:
            dummy["labels"] = self.input_ids()

        return dummy


class QuestionAnsweringGenerator(TextGenerator):
    def start_positions(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["sequence_length"],
            shape=(self.shapes["batch_size"],),
        )

    def end_positions(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["sequence_length"],
            shape=(self.shapes["batch_size"],),
        )

    def generate(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.with_labels:
            dummy["start_positions"] = self.start_positions()
            dummy["end_positions"] = self.end_positions()

        return dummy


class MaskedLanguageModelingGenerator(TextGenerator):
    def generate(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.with_labels:
            dummy["labels"] = self.input_ids()

        return dummy


class MultipleChoiceGenerator(TextGenerator):
    def input_ids(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["vocab_size"],
            shape=(
                self.shapes["batch_size"],
                self.shapes["num_choices"],
                self.shapes["sequence_length"],
            ),
        )

    def token_type_ids(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["type_vocab_size"],
            shape=(
                self.shapes["batch_size"],
                self.shapes["num_choices"],
                self.shapes["sequence_length"],
            ),
        )

    def attention_mask(self):
        return self.generate_random_integers(
            min_value=1,
            max_value=2,
            shape=(
                self.shapes["batch_size"],
                self.shapes["num_choices"],
                self.shapes["sequence_length"],
            ),
        )

    def labels(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_choices"],
            shape=(self.shapes["batch_size"],),
        )

    def generate(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["token_type_ids"] = self.token_type_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.with_labels:
            dummy["label"] = self.labels()

        return dummy


class ImageClassificationGenerator(ImageGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"],
            shape=(self.shapes["batch_size"],),
        )

    def generate(self):
        dummy = {}
        dummy["pixel_value"] = self.pixel_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class ObjectDetectionGenerator(ImageGenerator):
    def labels(self):
        return [
            {
                "class_labels": self.generate_random_integers(
                    min_value=0,
                    max_value=self.shapes["num_labels"],
                    shape=(self.shapes["num_queries"],),
                ),
                "boxes": self.generate_random_floats(
                    min_value=-1,
                    max_value=1,
                    shape=(self.shapes["num_queries"], 4),
                ),
            }
            for _ in range(self.shapes["batch_size"])
        ]

    def generate(self):
        dummy = {}
        dummy["pixel_value"] = self.pixel_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class SemanticSegmentationGenerator(ImageGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"],
            shape=(
                self.shapes["batch_size"],
                self.shapes["height"],
                self.shapes["width"],
            ),
        )

    def generate(self):
        dummy = {}
        dummy["pixel_values"] = self.pixel_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class AudioClassificationGenerator(AudioGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"],
            shape=(self.shapes["batch_size"],),
        )

    def generate(self):
        dummy = {}
        dummy["input_values"] = self.input_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class AutomaticSpeechRecognitionGenerator(AudioGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["vocab_size"],
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"],
            ),
        )

    def generate(self):
        dummy = {}
        dummy["input_values"] = self.input_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class PromptGenerator(TaskGenerator):
    def generate(self):
        dummy = {}

        dummy["prompt"] = ["prompt"] * self.shapes["batch_size"]

        return dummy


TASKS_TO_GENERATORS = {
    # model tasks
    "text-classification": TextClassificationGenerator,
    "token-classification": TokenClassificationGenerator,
    "text-generation": TextGenerationGenerator,
    "text2text-generation": TextGenerationGenerator,
    "question-answering": QuestionAnsweringGenerator,
    "fill-mask": MaskedLanguageModelingGenerator,
    "multiple-choice": MultipleChoiceGenerator,
    "image-classification": ImageClassificationGenerator,
    "object-detection": ObjectDetectionGenerator,
    "semantic-segmentation": SemanticSegmentationGenerator,
    # pipeline tasks
    "stable-diffusion": PromptGenerator,
    "stable-diffusion-xl": PromptGenerator,
}
