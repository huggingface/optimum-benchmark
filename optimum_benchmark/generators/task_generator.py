import logging
import random
import string
from abc import ABC
from typing import Dict, List, Tuple

# TODO: drop torch dependency and use numpy instead
import torch

LOGGER = logging.getLogger("generators")

DEFAULT_NUM_LABELS = 2
DEFAULT_VOCAB_SIZE = 2
DEFAULT_TYPE_VOCAB_SIZE = 2


class TaskGenerator(ABC):
    def __init__(self, shapes: Dict[str, int], with_labels: bool):
        self.shapes = shapes
        self.with_labels = with_labels

    @staticmethod
    def generate_constant_integers(value: int, shape: Tuple[int]):
        return torch.full(shape, value, dtype=torch.int64)

    @staticmethod
    def generate_constant_floats(value: float, shape: Tuple[int]):
        return torch.full(shape, value, dtype=torch.float32)

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
    def generate_random_strings(num_seq: int) -> List[str]:
        return [
            "".join(random.choice(string.ascii_letters + string.digits) for _ in range(random.randint(10, 100)))
            for _ in range(num_seq)
        ]

    def __call__(self):
        raise NotImplementedError("Generator must implement __call__ method")


class TextGenerator(TaskGenerator):
    def input_ids(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate input ids."
        )
        assert self.shapes.get("sequence_length", None) is not None, (
            "Sequence length must be provided, "
            "please provide it in `input_shapes` as `sequence_length` to be able to generate input ids."
        )

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("vocab_size", None) or DEFAULT_VOCAB_SIZE,
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def attention_mask(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate attention masks."
        )
        assert self.shapes.get("sequence_length", None) is not None, (
            "Sequence length must be provided, "
            "please provide it in `input_shapes` as `sequence_length` to be able to generate attention masks."
        )

        return self.generate_constant_integers(
            value=1,  # no sparsity
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def token_type_ids(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate token type ids."
        )
        assert self.shapes.get("sequence_length", None) is not None, (
            "Sequence length must be provided, "
            "please provide it in `input_shapes` as `sequence_length` to be able to generate token type ids."
        )

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("type_vocab_size", None) or DEFAULT_TYPE_VOCAB_SIZE,
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def position_ids(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate position ids."
        )
        assert self.shapes.get("sequence_length", None) is not None, (
            "Sequence length must be provided, "
            "please provide it in `input_shapes` as `sequence_length` to be able to generate position ids."
        )

        return self.generate_ranges(
            start=0,
            stop=self.shapes["sequence_length"],
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def requires_token_type_ids(self):
        return self.shapes.get("type_vocab_size", None) is not None and self.shapes["type_vocab_size"] > 1

    def requires_position_ids(self):
        return (
            self.shapes.get("max_position_embeddings", None) is not None and self.shapes["max_position_embeddings"] > 1
        )


class ImageGenerator(TaskGenerator):
    def pixel_values(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate pixel values."
        )
        assert self.shapes.get("num_channels", None) is not None, (
            "Number of channels couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `num_channels` to be able to generate pixel values."
        )
        assert self.shapes.get("height", None) is not None, (
            "Height couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `height` to be able to generate pixel values."
        )
        assert self.shapes.get("width", None) is not None, (
            "Width couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `width` to be able to generate pixel values."
        )

        return self.generate_random_floats(
            min_value=0,
            max_value=1,
            shape=(self.shapes["batch_size"], self.shapes["num_channels"], self.shapes["height"], self.shapes["width"]),
        )


class AudioGenerator(TaskGenerator):
    def input_values(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate input values."
        )
        assert self.shapes.get("sequence_length", None) is not None, (
            "Sequence length must be provided, "
            "please provide it in `input_shapes` as `sequence_length` to be able to generate input values."
        )
        return self.generate_random_floats(
            min_value=-1, max_value=1, shape=(self.shapes["batch_size"], self.shapes["sequence_length"])
        )

    def input_features(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate input features."
        )
        assert self.shapes.get("feature_size", None) is not None, (
            "Feature size couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `feature_size` to be able to generate input features."
        )
        assert self.shapes.get("nb_max_frames", None) is not None, (
            "Number of max frames couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `nb_max_frames` to be able to generate input features."
        )

        return self.generate_random_floats(
            min_value=-1,
            max_value=1,
            shape=(self.shapes["batch_size"], self.shapes["feature_size"], self.shapes["nb_max_frames"]),
        )


class TextClassificationGenerator(TextGenerator):
    def labels(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate labels."
        )

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"] or DEFAULT_NUM_LABELS,
            shape=(self.shapes["batch_size"],),
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
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate labels."
        )
        assert self.shapes.get("sequence_length", None) is not None, (
            "Sequence length must be provided, "
            "please provide it in `input_shapes` as `sequence_length` to be able to generate labels."
        )

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"] or DEFAULT_NUM_LABELS,
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
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate start positions."
        )
        assert self.shapes.get("sequence_length", None) is not None, (
            "Sequence length must be provided, "
            "please provide it in `input_shapes` as `sequence_length` to be able to generate start positions."
        )

        return self.generate_random_integers(
            min_value=0, max_value=self.shapes["sequence_length"], shape=(self.shapes["batch_size"],)
        )

    def end_positions(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate end positions."
        )
        assert self.shapes.get("sequence_length", None) is not None, (
            "Sequence length must be provided, "
            "please provide it in `input_shapes` as `sequence_length` to be able to generate end positions."
        )

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
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate labels."
        )
        assert self.shapes.get("num_choices", None) is not None, (
            "Number of choices must be provided, "
            "please provide it in `input_shapes` as `num_choices` to be able to generate labels."
        )

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
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate labels."
        )

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"] or DEFAULT_NUM_LABELS,
            shape=(self.shapes["batch_size"],),
        )

    def __call__(self):
        dummy = {}
        dummy["pixel_values"] = self.pixel_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class ObjectDetectionGenerator(ImageGenerator):
    def labels(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate labels."
        )
        assert self.shapes.get("num_queries", None) is not None, (
            "Number of queries must be provided, "
            "please provide it in `input_shapes` as `num_queries` to be able to generate labels."
        )

        return [
            {
                "class_labels": self.generate_random_integers(
                    min_value=0,
                    max_value=self.shapes["num_labels"] or DEFAULT_NUM_LABELS,
                    shape=(self.shapes["num_queries"],),
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
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate labels."
        )
        assert self.shapes.get("height", None) is not None, (
            "Height couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `height` to be able to generate labels."
        )
        assert self.shapes.get("width", None) is not None, (
            "Width couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `width` to be able to generate labels."
        )

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["num_labels"] or DEFAULT_NUM_LABELS,
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
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate labels."
        )
        assert self.shapes.get("num_labels", None) is not None, (
            "Number of labels couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `num_labels` to be able to generate labels."
        )

        return self.generate_random_integers(
            min_value=0, max_value=self.shapes["num_labels"] or DEFAULT_NUM_LABELS, shape=(self.shapes["batch_size"],)
        )

    def __call__(self):
        dummy = {}
        dummy["input_values"] = self.input_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class AutomaticSpeechRecognitionGenerator(AudioGenerator):
    def labels(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate labels."
        )
        assert self.shapes.get("sequence_length", None) is not None, (
            "Sequence length must be provided, "
            "please provide it in `input_shapes` as `sequence_length` to be able to generate labels."
        )
        assert self.shapes.get("num_labels", None) is not None, (
            "Number of labels couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `num_labels` to be able to generate labels."
        )

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["vocab_size"] or DEFAULT_TYPE_VOCAB_SIZE,
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
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate prompts."
        )

        return self.generate_random_strings(num_seq=self.shapes["batch_size"])

    def __call__(self):
        dummy = {}
        dummy["prompt"] = self.prompt()

        return dummy


class FeatureExtractionGenerator(TextGenerator, ImageGenerator):
    def __call__(self):
        dummy = {}

        if self.shapes.get("num_channels", None) is not None and self.shapes.get("height", None) is not None:
            dummy["pixel_values"] = self.pixel_values()
        else:
            dummy["input_ids"] = self.input_ids()
            dummy["attention_mask"] = self.attention_mask()

            if self.requires_token_type_ids():
                dummy["token_type_ids"] = self.token_type_ids()

            if self.requires_position_ids():
                dummy["position_ids"] = self.position_ids()

        return dummy


class ImageTextToTextGenerationGenerator(TextGenerator, ImageGenerator):
    def input_ids(self):
        assert self.shapes.get("batch_size", None) is not None, (
            "Batch size must be provided, "
            "please provide it in `input_shapes` as `batch_size` to be able to generate input ids."
        )
        assert self.shapes.get("sequence_length", None) is not None, (
            "Sequence length must be provided, "
            "please provide it in `input_shapes` as `sequence_length` to be able to generate input ids."
        )
        assert self.shapes.get("num_images", None) is not None, (
            "Number of images must be provided, "
            "please provide it in `input_shapes` as `num_images` to be able to generate input ids."
        )
        assert self.shapes.get("num_channels", None) is not None, (
            "Number of channels couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `num_channels` to be able to generate input ids."
        )
        assert self.shapes.get("height", None) is not None, (
            "Height couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `height` to be able to generate input ids."
        )
        assert self.shapes.get("width", None) is not None, (
            "Width couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `width` to be able to generate input ids."
        )
        assert self.shapes.get("patch_size", None) is not None, (
            "Patch size must be provided, "
            "please provide it in `input_shapes` as `patch_size` to be able to generate input ids."
        )
        assert self.shapes.get("temporal_patch_size", None) is not None, (
            "Temporal patch size must be provided, "
            "please provide it in `input_shapes` as `temporal_patch_size` to be able to generate input ids."
        )
        assert self.shapes.get("spatial_merge_size", None) is not None, (
            "Spatial merge size must be provided, "
            "please provide it in `input_shapes` as `spatial_merge_size` to be able to generate input ids."
        )
        assert self.shapes.get("image_token_id", None) is not None, (
            "Image token id must be provided, "
            "please provide it in `input_shapes` as `image_token_id` to be able to generate input ids."
        )

        text_tokens = self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("vocab_size", None) or DEFAULT_VOCAB_SIZE,
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"],
            ),
        )
        image_tokens = self.generate_constant_integers(
            value=self.shapes["image_token_id"],
            shape=(
                self.shapes["batch_size"],
                int(
                    self.shapes["num_images"]
                    * self.shapes["height"]
                    * self.shapes["width"]
                    / self.shapes["temporal_patch_size"]
                    / self.shapes["spatial_merge_size"]
                    / self.shapes["patch_size"] ** 2
                ),
            ),
        )

        return torch.cat((text_tokens, image_tokens), dim=1)

    def pixel_values(self):
        assert self.shapes.get("num_images", None) is not None, (
            "Number of images must be provided, "
            "please provide it in `input_shapes` as `num_images` to be able to generate pixel values."
        )
        assert self.shapes.get("num_channels", None) is not None, (
            "Number of channels couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `num_channels` to be able to generate pixel values."
        )
        assert self.shapes.get("height", None) is not None, (
            "Height couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `height` to be able to generate pixel values."
        )
        assert self.shapes.get("width", None) is not None, (
            "Width couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `width` to be able to generate pixel values."
        )
        assert self.shapes.get("patch_size", None) is not None, (
            "Patch size must be provided, "
            "please provide it in `input_shapes` as `patch_size` to be able to generate pixel values."
        )
        assert self.shapes.get("temporal_patch_size", None) is not None, (
            "Temporal patch size must be provided, "
            "please provide it in `input_shapes` as `temporal_patch_size` to be able to generate pixel values."
        )

        return self.generate_random_floats(
            min_value=0,
            max_value=1,
            shape=(
                self.shapes["num_images"]
                * int(self.shapes["height"] / self.shapes["patch_size"])
                * int(self.shapes["width"] / self.shapes["patch_size"]),
                self.shapes["num_channels"]
                * self.shapes["patch_size"]
                * self.shapes["patch_size"]
                * self.shapes["temporal_patch_size"],
            ),
        )

    def image_grid_thw(self):
        assert self.shapes.get("num_images", None) is not None, (
            "Number of images must be provided, "
            "please provide it in `input_shapes` as `num_images` to be able to generate image grid."
        )
        assert self.shapes.get("height", None) is not None, (
            "Height couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `height` to be able to generate image grid."
        )
        assert self.shapes.get("width", None) is not None, (
            "Width couldn't be inferred automatically from model, "
            "please provide it in `input_shapes` as `width` to be able to generate image grid."
        )

        return torch.tensor(
            [
                [
                    self.shapes["num_images"],
                    int(self.shapes["height"] / self.shapes["patch_size"]),
                    int(self.shapes["width"] / self.shapes["patch_size"]),
                ]
            ]
        )

    def __call__(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["pixel_values"] = self.pixel_values()
        dummy["image_grid_thw"] = self.image_grid_thw()

        print("input_ids", dummy["input_ids"].shape)
        print("pixel_values", dummy["pixel_values"].shape)
        print("image_grid_thw", dummy["image_grid_thw"].shape)

        if self.with_labels:
            dummy["labels"] = self.input_ids()

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
    "image-text-to-text": ImageTextToTextGenerationGenerator,
    # diffusers pipelines tasks
    "text-to-image": PromptGenerator,
    "stable-diffusion": PromptGenerator,
    "stable-diffusion-xl": PromptGenerator,
}
