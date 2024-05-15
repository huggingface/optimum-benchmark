import random
import string
from abc import ABC
from typing import Tuple

# TODO: drop torch dependency and use numpy instead ?
import torch


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

    @staticmethod
    def generate_ranges(start: int, stop: int, shape: Tuple[int]):
        return torch.arange(start, stop).repeat(shape[0], 1)

    @staticmethod
    def generate_random_strings(shape: Tuple[int]):
        return [
            "".join(random.choice(string.ascii_letters + string.digits) for _ in range(shape[1]))
            for _ in range(shape[0])
        ]

    def __call__(self):
        raise NotImplementedError("Generator must implement __call__ method")


class TextGenerator(TaskGenerator):
    def input_ids(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["vocab_size"],
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def attention_mask(self):
        return self.generate_random_integers(
            min_value=1,  # avoid sparse attention
            max_value=2,
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def token_type_ids(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["type_vocab_size"],
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def position_ids(self):
        return self.generate_ranges(
            start=0,
            stop=self.shapes["sequence_length"],
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def requires_token_type_ids(self):
        return self.shapes["type_vocab_size"] is not None and self.shapes["type_vocab_size"] > 1

    def requires_position_ids(self):
        return self.shapes["max_position_embeddings"] is not None


class ImageGenerator(TaskGenerator):
    def pixel_values(self):
        return self.generate_random_floats(
            min_value=0,
            max_value=1,
            shape=(self.shapes["batch_size"], self.shapes["num_channels"], self.shapes["height"], self.shapes["width"]),
        )


class AudioGenerator(TaskGenerator):
    def input_values(self):
        return self.generate_random_floats(
            min_value=-1, max_value=1, shape=(self.shapes["batch_size"], self.shapes["sequence_length"])
        )

    def input_features(self):
        return self.generate_random_floats(
            min_value=-1,
            max_value=1,
            shape=(self.shapes["batch_size"], self.shapes["feature_size"], self.shapes["nb_max_frames"]),
        )


class TextClassificationGenerator(TextGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0, max_value=self.shapes["num_labels"], shape=(self.shapes["batch_size"],)
        )

    def __call__(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.requires_token_type_ids():
            dummy["token_type_ids"] = self.token_type_ids()

        if self.requires_position_ids():
            dummy["position_ids"] = self.position_ids()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class TokenClassificationGenerator(TextGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"],
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def __call__(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.requires_token_type_ids():
            dummy["token_type_ids"] = self.token_type_ids()

        if self.requires_position_ids():
            dummy["position_ids"] = self.position_ids()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class TextGenerationGenerator(TextGenerator):
    def __call__(self):
        dummy = {}
        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.with_labels:
            dummy["labels"] = self.input_ids()

        return dummy


class Text2TextGenerationGenerator(TextGenerator):
    def __call__(self):
        dummy = {}
        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.with_labels:
            dummy["labels"] = self.input_ids()

        return dummy


class QuestionAnsweringGenerator(TextGenerator):
    def start_positions(self):
        return self.generate_random_integers(
            min_value=0, max_value=self.shapes["sequence_length"], shape=(self.shapes["batch_size"],)
        )

    def end_positions(self):
        return self.generate_random_integers(
            min_value=0, max_value=self.shapes["sequence_length"], shape=(self.shapes["batch_size"],)
        )

    def __call__(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()
        dummy["token_type_ids"] = self.token_type_ids()

        if self.with_labels:
            dummy["start_positions"] = self.start_positions()
            dummy["end_positions"] = self.end_positions()

        return dummy


class MaskedLanguageModelingGenerator(TextGenerator):
    def __call__(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.requires_token_type_ids():
            dummy["token_type_ids"] = self.token_type_ids()

        if self.requires_position_ids():
            dummy["position_ids"] = self.position_ids()

        if self.with_labels:
            dummy["labels"] = self.input_ids()

        return dummy


class MultipleChoiceGenerator(TextGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0, max_value=self.shapes["num_choices"], shape=(self.shapes["batch_size"],)
        )

    def __call__(self):
        dummy = {}

        dummy["input_ids"] = (
            self.input_ids()
            .reshape(self.shapes["batch_size"], 1, self.shapes["sequence_length"])
            .repeat(1, self.shapes["num_choices"], 1)
        )

        dummy["attention_mask"] = (
            self.attention_mask()
            .reshape(self.shapes["batch_size"], 1, self.shapes["sequence_length"])
            .repeat(1, self.shapes["num_choices"], 1)
        )

        if self.requires_token_type_ids():
            dummy["token_type_ids"] = (
                self.token_type_ids()
                .reshape(self.shapes["batch_size"], 1, self.shapes["sequence_length"])
                .repeat(1, self.shapes["num_choices"], 1)
            )

        if self.with_labels:
            dummy["label"] = self.labels()

        return dummy


class ImageClassificationGenerator(ImageGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0, max_value=self.shapes["num_labels"], shape=(self.shapes["batch_size"],)
        )

    def __call__(self):
        dummy = {}
        dummy["pixel_values"] = self.pixel_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class ObjectDetectionGenerator(ImageGenerator):
    def labels(self):
        return [
            {
                "class_labels": self.generate_random_integers(
                    min_value=0, max_value=self.shapes["num_labels"], shape=(self.shapes["num_queries"],)
                ),
                "boxes": self.generate_random_floats(min_value=-1, max_value=1, shape=(self.shapes["num_queries"], 4)),
            }
            for _ in range(self.shapes["batch_size"])
        ]

    def __call__(self):
        dummy = {}
        dummy["pixel_values"] = self.pixel_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class SemanticSegmentationGenerator(ImageGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"],
            shape=(self.shapes["batch_size"], self.shapes["height"], self.shapes["width"]),
        )

    def __call__(self):
        dummy = {}
        dummy["pixel_values"] = self.pixel_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class AudioClassificationGenerator(AudioGenerator):
    def labels(self):
        return self.generate_random_integers(
            min_value=0, max_value=self.shapes["num_labels"], shape=(self.shapes["batch_size"],)
        )

    def __call__(self):
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
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def __call__(self):
        dummy = {}
        dummy["input_values"] = self.input_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class PromptGenerator(TaskGenerator):
    def prompt(self):
        return self.generate_random_strings(shape=(self.shapes["batch_size"], 10))

    def __call__(self):
        dummy = {}
        dummy["prompt"] = self.prompt()

        return dummy


class FeatureExtractionGenerator(TextGenerator, ImageGenerator):
    def __call__(self):
        dummy = {}

        if self.shapes["num_channels"] is not None and self.shapes["height"] is not None:
            dummy["pixel_values"] = self.pixel_values()
        else:
            dummy["input_ids"] = self.input_ids()
            dummy["attention_mask"] = self.attention_mask()

            if self.requires_token_type_ids():
                dummy["token_type_ids"] = self.token_type_ids()

            if self.requires_position_ids():
                dummy["position_ids"] = self.position_ids()

        return dummy


TASKS_TO_GENERATORS = {
    # transformers models tasks
    "feature-extraction": FeatureExtractionGenerator,
    "text-classification": TextClassificationGenerator,
    "token-classification": TokenClassificationGenerator,
    "text-generation": TextGenerationGenerator,
    "text2text-generation": Text2TextGenerationGenerator,
    "question-answering": QuestionAnsweringGenerator,
    "fill-mask": MaskedLanguageModelingGenerator,
    "multiple-choice": MultipleChoiceGenerator,
    "image-classification": ImageClassificationGenerator,
    "object-detection": ObjectDetectionGenerator,
    "semantic-segmentation": SemanticSegmentationGenerator,
    # diffusers pipelines tasks
    "text-to-image": PromptGenerator,
    "stable-diffusion": PromptGenerator,
    "stable-diffusion-xl": PromptGenerator,
}
