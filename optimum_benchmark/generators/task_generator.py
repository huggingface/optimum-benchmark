import logging

from .base import BaseGenerator

LOGGER = logging.getLogger("generators")

DEFAULT_NUM_LABELS = 2
DEFAULT_VOCAB_SIZE = 2
DEFAULT_TYPE_VOCAB_SIZE = 2


class TextGenerator(BaseGenerator):
    def input_ids(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length"])

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("vocab_size", DEFAULT_VOCAB_SIZE),
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def attention_mask(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length"])

        return self.generate_constant_integers(
            value=1,  # no sparsity
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def token_type_ids(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length"])

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("type_vocab_size", DEFAULT_TYPE_VOCAB_SIZE),
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

    def position_ids(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length"])

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


class ImageGenerator(BaseGenerator):
    def pixel_values(self):
        self.assert_not_missing_shapes(["batch_size", "num_channels", "height", "width"])

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


class AudioGenerator(BaseGenerator):
    def input_values(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length"])

        return self.generate_random_floats(
            min_value=-1,
            max_value=1,
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"],
            ),
        )

    def input_features(self):
        self.assert_not_missing_shapes(["batch_size", "feature_size", "nb_max_frames"])

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
        self.assert_not_missing_shapes(["batch_size"])

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("num_labels", DEFAULT_NUM_LABELS),
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
        self.assert_not_missing_shapes(["batch_size", "sequence_length"])

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("num_labels", DEFAULT_NUM_LABELS),
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
        self.assert_not_missing_shapes(["batch_size", "sequence_length"])

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["sequence_length"],
            shape=(self.shapes["batch_size"],),
        )

    def end_positions(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length"])

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes["sequence_length"],
            shape=(self.shapes["batch_size"],),
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
    def input_ids(self):
        self.assert_not_missing_shapes(["batch_size", "num_choices", "sequence_length"])

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("vocab_size", DEFAULT_VOCAB_SIZE),
            shape=(self.shapes["batch_size"], self.shapes["num_choices"], self.shapes["sequence_length"]),
        )

    def attention_mask(self):
        self.assert_not_missing_shapes(["batch_size", "num_choices", "sequence_length"])

        return self.generate_constant_integers(
            value=1,  # no sparsity
            shape=(self.shapes["batch_size"], self.shapes["num_choices"], self.shapes["sequence_length"]),
        )

    def token_type_ids(self):
        self.assert_not_missing_shapes(["batch_size", "num_choices", "sequence_length"])

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("type_vocab_size", DEFAULT_TYPE_VOCAB_SIZE),
            shape=(self.shapes["batch_size"], self.shapes["num_choices"], self.shapes["sequence_length"]),
        )

    def labels(self):
        self.assert_not_missing_shapes(["batch_size", "num_choices"])

        return self.generate_random_integers(
            min_value=0, max_value=self.shapes["num_choices"], shape=(self.shapes["batch_size"],)
        )

    def __call__(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()

        if self.requires_token_type_ids():
            dummy["token_type_ids"] = self.token_type_ids()

        if self.with_labels:
            dummy["label"] = self.labels()

        return dummy


class ImageClassificationGenerator(ImageGenerator):
    def labels(self):
        self.assert_not_missing_shapes(["batch_size"])

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("num_labels", DEFAULT_NUM_LABELS),
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
        self.assert_not_missing_shapes(["batch_size", "num_queries"])

        return [
            {
                "class_labels": self.generate_random_integers(
                    min_value=0,
                    max_value=self.shapes.get("num_labels", DEFAULT_NUM_LABELS),
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
        self.assert_not_missing_shapes(["batch_size", "height", "width"])

        return self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("num_labels", DEFAULT_NUM_LABELS),
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
        self.assert_not_missing_shapes(["batch_size"])

        return self.generate_random_integers(
            min_value=0, max_value=self.shapes.get("num_labels", DEFAULT_NUM_LABELS), shape=(self.shapes["batch_size"],)
        )

    def __call__(self):
        dummy = {}
        dummy["input_values"] = self.input_values()

        if self.with_labels:
            dummy["labels"] = self.labels()

        return dummy


class AutomaticSpeechRecognitionGenerator(AudioGenerator):
    def labels(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length"])

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


class PromptGenerator(BaseGenerator):
    def prompt(self):
        self.assert_not_missing_shapes(["batch_size"])

        return self.generate_random_strings(num_seq=self.shapes["batch_size"])

    def __call__(self):
        dummy = {}
        dummy["prompt"] = self.prompt()

        return dummy


class FeatureExtractionGenerator(TextGenerator, ImageGenerator):
    def __call__(self):
        dummy = {}

        if self.shapes.get("sequence_length", None) is not None:
            dummy["input_ids"] = self.input_ids()
            dummy["attention_mask"] = self.attention_mask()

            if self.requires_token_type_ids():
                dummy["token_type_ids"] = self.token_type_ids()

            if self.requires_position_ids():
                dummy["position_ids"] = self.position_ids()

        if self.shapes.get("height", None) is not None:
            dummy["pixel_values"] = self.pixel_values()

        return dummy


class ImageTextToTextGenerator(TextGenerator, ImageGenerator):
    def __call__(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["attention_mask"] = self.attention_mask()
        dummy["pixel_values"] = self.pixel_values()

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
    "image-text-to-text": ImageTextToTextGenerator,
    # diffusers pipelines tasks
    "text-to-image": PromptGenerator,
}
