import logging

import torch

from .base import BaseGenerator

LOGGER = logging.getLogger("generators")

DEFAULT_VOCAB_SIZE = 2


class IdeficsGenerator(BaseGenerator):
    def input_ids(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length", "num_images", "image_token_id"])

        text_tokens = self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("vocab_size", DEFAULT_VOCAB_SIZE),
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

        image_tokens = self.generate_constant_integers(
            value=self.shapes["image_token_id"],
            shape=(self.shapes["batch_size"], self.shapes["num_images"]),
        )

        return torch.cat((text_tokens, image_tokens), dim=1)

    def attention_mask(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length", "num_images"])

        return self.generate_constant_integers(
            value=1,  # no sparsity
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"] + self.shapes["num_images"],
            ),
        )

    def pixel_values(self):
        self.assert_not_missing_shapes(["batch_size", "num_images", "num_channels", "height", "width"])

        return self.generate_random_floats(
            min_value=0,
            max_value=1,
            shape=(
                self.shapes["batch_size"],
                self.shapes["num_images"],
                self.shapes["num_channels"],
                self.shapes["height"],
                self.shapes["width"],
            ),
        )

    def image_attention_mask(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length", "num_images"])

        return self.generate_constant_integers(
            value=1,  # no sparsity
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"] + self.shapes["num_images"],
                self.shapes["num_images"],
            ),
        )

    def __call__(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["pixel_values"] = self.pixel_values()
        dummy["attention_mask"] = self.attention_mask()
        dummy["image_attention_mask"] = self.image_attention_mask()

        if self.with_labels:
            dummy["labels"] = self.input_ids()

        return dummy


class Idefics2Generator(BaseGenerator):
    def input_ids(self):
        self.assert_not_missing_shapes(
            ["batch_size", "sequence_length", "num_images", "image_seq_len", "image_token_id", "do_image_splitting"]
        )

        text_tokens = self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("vocab_size", DEFAULT_VOCAB_SIZE),
            shape=(self.shapes["batch_size"], self.shapes["sequence_length"]),
        )

        image_tokens = self.generate_constant_integers(
            value=self.shapes["image_token_id"],
            shape=(
                self.shapes["batch_size"],
                self.shapes["num_images"]
                * self.shapes["image_seq_len"]
                * (5 if self.shapes["do_image_splitting"] else 1),
            ),
        )

        return torch.cat((text_tokens, image_tokens), dim=1)

    def attention_mask(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length", "num_images", "do_image_splitting"])

        return self.generate_constant_integers(
            value=1,  # no sparsity
            shape=(
                self.shapes["batch_size"],
                self.shapes["sequence_length"]
                + self.shapes["num_images"]
                * self.shapes["image_seq_len"]
                * (5 if self.shapes["do_image_splitting"] else 1),
            ),
        )

    def pixel_values(self):
        self.assert_not_missing_shapes(
            ["batch_size", "num_images", "num_channels", "height", "width", "do_image_splitting"]
        )

        return self.generate_random_floats(
            min_value=0,
            max_value=1,
            shape=(
                self.shapes["batch_size"],
                self.shapes["num_images"] * (5 if self.shapes["do_image_splitting"] else 1),
                self.shapes["num_channels"],
                self.shapes["height"],
                self.shapes["width"],
            ),
        )

    def pixel_attention_mask(self):
        self.assert_not_missing_shapes(["batch_size", "sequence_length", "num_images", "do_image_splitting"])

        return self.generate_constant_integers(
            value=1,  # no sparsity
            shape=(
                self.shapes["batch_size"],
                self.shapes["num_images"] * (5 if self.shapes["do_image_splitting"] else 1),
                self.shapes["height"],
                self.shapes["width"],
            ),
        )

    def __call__(self):
        dummy = {}

        dummy["input_ids"] = self.input_ids()
        dummy["pixel_values"] = self.pixel_values()
        dummy["attention_mask"] = self.attention_mask()
        dummy["pixel_attention_mask"] = self.pixel_attention_mask()

        print("input_ids", dummy["input_ids"].shape)
        print("pixel_values", dummy["pixel_values"].shape)
        print("attention_mask", dummy["attention_mask"].shape)
        print("pixel_attention_mask", dummy["pixel_attention_mask"].shape)

        if self.with_labels:
            dummy["labels"] = self.input_ids()

        return dummy


class Qwen2VLGenerator(BaseGenerator):
    def input_ids(self):
        self.assert_not_missing_shapes(
            [
                "batch_size",
                "sequence_length",
                "num_images",
                "num_channels",
                "height",
                "width",
                "patch_size",
                "temporal_patch_size",
                "spatial_merge_size",
                "image_token_id",
            ]
        )

        text_tokens = self.generate_random_integers(
            min_value=0,
            max_value=self.shapes.get("vocab_size", DEFAULT_VOCAB_SIZE),
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
        self.assert_not_missing_shapes(
            ["num_images", "num_channels", "height", "width", "patch_size", "temporal_patch_size"]
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
        self.assert_not_missing_shapes(["num_images", "height", "width", "patch_size"])

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

        if self.with_labels:
            dummy["labels"] = self.input_ids()

        return dummy


MODEL_TYPE_TO_GENERATORS = {
    "idefics": IdeficsGenerator,
    "idefics2": Idefics2Generator,
    "qwen2_vl": Qwen2VLGenerator,
}
