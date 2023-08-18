from optimum_benchmark.generators.utils import *


class TaskGenerator:
    def __init__(self, dummy_shapes, model_shapes):
        self.dummy_shapes = dummy_shapes
        self.model_shapes = model_shapes

    def generate(self, mode: str, with_labels: bool):
        raise NotImplementedError

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
        }


class TextClassificationGenerator(TaskGenerator):
    def generate(self, mode: str, with_labels: bool):
        dummy = {}
        dummy["input_ids"] = generate_input_ids(
            vocab_size=self.model_shapes["vocab_size"],
            batch_size=self.dummy_shapes["batch_size"],
            sequence_length=self.dummy_shapes["sequence_length"],
        )

        dummy["attention_mask"] = generate_attention_mask(
            input_ids_or_values=dummy["input_ids"],
        )

        dummy["token_type_ids"] = generate_token_type_ids(
            batch_size=self.dummy_shapes["batch_size"],
            sequence_length=self.dummy_shapes["sequence_length"],
        )

        if with_labels:
            dummy["labels"] = generate_sequence_labels(
                batch_size=self.dummy_shapes["batch_size"],
                # num_labels in this case is not available in the config
                # but ratherthe moment you instantiate the text classification model
                # we can't access it here, so we just use the default value
                # which is 2 which works for binary and multi-class classification
            )

        return dummy

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
            "sequence_length": self.dummy_shapes["sequence_length"],
        }


class TokenClassificationGenerator(TaskGenerator):
    def generate(self, mode: str, with_labels: bool):
        dummy = {}
        dummy["input_ids"] = generate_input_ids(
            vocab_size=self.model_shapes["vocab_size"],
            batch_size=self.dummy_shapes["batch_size"],
            sequence_length=self.dummy_shapes["sequence_length"],
        )
        dummy["attention_mask"] = generate_attention_mask(
            input_ids_or_values=dummy["input_ids"],
        )
        if with_labels:
            dummy["labels"] = generate_token_labels(
                num_labels=self.model_shapes["num_labels"],
                batch_size=self.dummy_shapes["batch_size"],
                sequence_length=self.dummy_shapes["sequence_length"],
            )

        return dummy

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
            "sequence_length": self.dummy_shapes["sequence_length"],
        }


class TextGenerationGenerator(TaskGenerator):
    def generate(self, mode: str, with_labels: bool):
        dummy = {}
        dummy["input_ids"] = generate_input_ids(
            vocab_size=self.model_shapes["vocab_size"],
            batch_size=self.dummy_shapes["batch_size"],
            sequence_length=self.dummy_shapes["sequence_length"],
        )
        dummy["attention_mask"] = generate_attention_mask(
            input_ids_or_values=dummy["input_ids"],
        )

        if with_labels:
            dummy["labels"] = dummy["input_ids"]

        return dummy

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
            "sequence_length": self.dummy_shapes["sequence_length"],
        }


class QuestionAnsweringGenerator(TaskGenerator):
    def generate(self, mode: str, with_labels: bool):
        dummy = {}
        dummy["input_ids"] = generate_input_ids(
            vocab_size=self.model_shapes["vocab_size"],
            batch_size=self.dummy_shapes["batch_size"],
            sequence_length=self.dummy_shapes["sequence_length"],
        )
        dummy["attention_mask"] = generate_attention_mask(
            input_ids_or_values=dummy["input_ids"],
        )

        if with_labels:
            dummy["start_positions"] = generate_start_positions(
                batch_size=self.dummy_shapes["batch_size"],
            )
            dummy["end_positions"] = generate_end_positions(
                batch_size=self.dummy_shapes["batch_size"],
                sequence_length=self.dummy_shapes["sequence_length"],
            )

        return dummy

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
            "sequence_length": self.dummy_shapes["sequence_length"],
        }


class MaskedLanguageModelingGenerator(TaskGenerator):
    def generate(self, mode: str, with_labels: bool):
        dummy = {}
        dummy["input_ids"] = generate_input_ids(
            vocab_size=self.model_shapes["vocab_size"],
            batch_size=self.dummy_shapes["batch_size"],
            sequence_length=self.dummy_shapes["sequence_length"],
        )
        dummy["attention_mask"] = generate_attention_mask(
            input_ids_or_values=dummy["input_ids"],
        )

        if with_labels:
            dummy["labels"] = dummy["input_ids"]

        return dummy

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
            "sequence_length": self.dummy_shapes["sequence_length"],
        }


class MultipleChoiceGenerator(TaskGenerator):
    def generate(self, mode: str, with_labels: bool):
        dummy = {}
        dummy["input_ids"] = generate_multiple_choice_input_ids(
            vocab_size=self.model_shapes["vocab_size"],
            batch_size=self.dummy_shapes["batch_size"],
            sequence_length=self.dummy_shapes["sequence_length"],
            num_choices=self.dummy_shapes["num_choices"],
        )
        dummy["token_type_ids"] = generate_multiple_choice_token_type_ids(
            batch_size=self.dummy_shapes["batch_size"],
            sequence_length=self.dummy_shapes["sequence_length"],
            num_choices=self.dummy_shapes["num_choices"],
        )
        dummy["attention_mask"] = generate_attention_mask(
            input_ids_or_values=dummy["input_ids"],
        )

        if with_labels:
            dummy["label"] = generate_multiple_choice_labels(
                num_choices=self.dummy_shapes["num_choices"],
                batch_size=self.dummy_shapes["batch_size"],
            )

        return dummy

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
            "sequence_length": self.dummy_shapes["sequence_length"],
            "num_choices": self.dummy_shapes["num_choices"],
        }


class ImageClassificationGenerator(TaskGenerator):
    def generate(self, mode: str, with_labels: bool):
        dummy = {}
        dummy["pixel_value"] = (
            generate_pixel_values(
                batch_size=self.dummy_shapes["batch_size"],
                num_channels=self.model_shapes["num_channels"],
                height=self.model_shapes["height"],
                width=self.model_shapes["width"],
            )
            / 255
        )

        if with_labels:
            dummy["labels"] = generate_sequence_labels(
                num_labels=self.model_shapes["num_labels"],
                batch_size=self.dummy_shapes["batch_size"],
            )

        return dummy

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
            "num_channels": self.model_shapes["num_channels"],
            "height": self.model_shapes["height"],
            "width": self.model_shapes["width"],
        }


class ObjectDetectionGenerator(TaskGenerator):
    def generate(self, mode: str, with_labels: bool):
        dummy = {}
        dummy["pixel_value"] = (
            generate_pixel_values(
                batch_size=self.dummy_shapes["batch_size"],
                num_channels=self.model_shapes["num_channels"],
                height=self.model_shapes["height"],
                width=self.model_shapes["width"],
            )
            / 255
        )

        if with_labels:
            dummy["labels"] = generate_object_detection_labels(
                batch_size=self.dummy_shapes["batch_size"],
                num_labels=self.model_shapes["num_labels"],
                num_queries=self.model_shapes["num_queries"],
            )

        return dummy

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
            "num_channels": self.model_shapes["num_channels"],
            "height": self.model_shapes["height"],
            "width": self.model_shapes["width"],
            "num_queries": self.model_shapes["num_queries"],
        }


class SemanticSegmentationGenerator(TaskGenerator):
    def generate(self, mode: str, with_labels: bool):
        dummy = {}
        dummy["pixel_values"] = (
            generate_pixel_values(
                batch_size=self.dummy_shapes["batch_size"],
                num_channels=self.model_shapes["num_channels"],
                height=self.model_shapes["height"],
                width=self.model_shapes["width"],
            )
            / 255
        )

        if with_labels:
            dummy["labels"] = generate_semantic_segmentation_labels(
                batch_size=self.dummy_shapes["batch_size"],
                height=self.model_shapes["height"],
                width=self.model_shapes["width"],
                num_labels=self.model_shapes["num_labels"],
            )

        return dummy

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
            "num_channels": self.model_shapes["num_channels"],
            "height": self.model_shapes["height"],
            "width": self.model_shapes["width"],
        }


class PromptGenerator(TaskGenerator):
    def generate(self, mode: str, with_labels: bool):
        dummy = {}
        dummy["prompt"] = generate_prompt(
            batch_size=self.dummy_shapes["batch_size"],
        )

        return dummy

    def get_static_shapes(self):
        return {
            "batch_size": self.dummy_shapes["batch_size"],
            "height": -1,  # this one is tricky
            "width": -1,  # this one is tricky
            "num_images_per_prompt": -1,  # this one is tricky
        }


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
