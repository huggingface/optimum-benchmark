from typing import Dict

from datasets import Dataset
from PIL.Image import Image
from transformers import PreTrainedTokenizer
import numpy as np

from ...backends.transformers_utils import PretrainedProcessor, PretrainedConfig
from .config import EnergyStarConfig


def preprocess(dataset: Dataset, task: str, config: EnergyStarConfig, preprocessor: PretrainedProcessor,  pretrained_config: PretrainedConfig) -> Dataset:
    task_to_preprocessing = {
        "feature-extraction": feature_extraction_preprocessing,
        "sentence-similarity": sentence_similarity_preprocessing,
        "text-classification": text_classification_preprocessing,
        "question-answering": question_answering_preprocessing,
        "text-generation": text_generation_preprocessing,
        "text2text-generation": text2text_generation_preprocessing,
        "summarization": summarization_preprocessing,
        "stable-diffusion": image_generation_preprocessing,
        "automatic-speech-recognition": automatic_speech_recognition_preprocessing,
        "image-to-text": image_to_text_preprocessing,
        "image-classification": image_preprocessing,
        "object-detection": image_preprocessing,
    }

    return task_to_preprocessing[task](dataset, config, preprocessor, pretrained_config)


def feature_extraction_preprocessing(
    dataset: Dataset, config: EnergyStarConfig, tokenizer: PreTrainedTokenizer,  pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(lambda example: example[config.text_column_name] != "")

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))

    # Add a pad token if the tokenizer doesn't have one
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding = False if config.input_shapes["batch_size"] == 1 else True

    def tokenize_function(examples):
        return tokenizer(
            examples[config.text_column_name],
            padding=padding,
            truncation=config.truncation,
            max_length = getattr(pretrained_config, "max_position_embeddings", 512)
            )


    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.features,
        desc="Running tokenizer on dataset",
    ).with_format(
        "torch"
    )  # We don't want a torch dependency here but for now the only backend used for this benchmark is PyTorch

    return dataset

def summarization_preprocessing(
    dataset: Dataset, config: EnergyStarConfig, tokenizer: PreTrainedTokenizer,  pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(lambda example: example[config.text_column_name] != "")

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))

    # Add a pad token if the tokenizer doesn't have one
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding = False if config.input_shapes["batch_size"] == 1 else True

    def tokenize_function(examples):
        return tokenizer(
            examples[config.text_column_name],
            padding=padding,
            truncation=config.truncation,
            max_length = getattr(pretrained_config, "max_position_embeddings", 512)
            )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.features,
        desc="Running tokenizer on dataset",
    ).with_format("torch")

    return dataset

def text_classification_preprocessing(
    dataset: Dataset, config: EnergyStarConfig, tokenizer: PreTrainedTokenizer, pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(lambda example: example[config.text_column_name] != "")

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))

    # Add a pad token if the tokenizer doesn't have one
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding = False if config.input_shapes["batch_size"] == 1 else True

    def tokenize_function(examples):
        return tokenizer(
            examples[config.text_column_name],
            padding=padding,
            truncation=config.truncation,
            max_length = getattr(pretrained_config, "max_position_embeddings", 512)
            )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.features,
        desc="Running tokenizer on dataset",
    ).with_format("torch")

    return dataset

def image_generation_preprocessing(
    dataset: Dataset, config: EnergyStarConfig, processor: PretrainedProcessor,  pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(lambda example: example[config.text_column_name] != "")

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))

    return dataset


def question_answering_preprocessing(
    dataset: Dataset, config: EnergyStarConfig, tokenizer: PreTrainedTokenizer,  pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(
            lambda example: (example[config.question_column_name], example[config.context_column_name]) != ""
        )

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))

    # Add a pad token if the tokenizer doesn't have one
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding = False if config.input_shapes["batch_size"] == 1 else True

    def tokenize_function(examples):
        return tokenizer(
            examples[config.question_column_name],
            examples[config.context_column_name],
            padding=padding,
            truncation=config.truncation,
            max_length = getattr(pretrained_config, "max_position_embeddings", 512)
            )


    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.features,
        desc="Running tokenizer on dataset",
    ).with_format("torch")

    return dataset

def text2text_generation_preprocessing(
    dataset: Dataset, config: EnergyStarConfig, tokenizer: PreTrainedTokenizer,  pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(lambda example: example[config.text_column_name] != "")

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))

    # Add a pad token if the tokenizer doesn't have one
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding = False if config.input_shapes["batch_size"] == 1 else True


    def tokenize_function(examples):
        examples[config.text_column_name] = config.dataset_prefix + examples[config.text_column_name]
        return tokenizer(
            examples[config.text_column_name],
            truncation=config.truncation,
            return_token_type_ids=False,
            padding=padding,
            max_length = getattr(pretrained_config, "max_position_embeddings", 512)- len(tokenizer(config.dataset_prefix))
            )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        writer_batch_size=50,
        remove_columns=dataset.features,
        desc="Running tokenizer on dataset",
    ).with_format("torch")

    return dataset

def text_generation_preprocessing(
    dataset: Dataset, config: EnergyStarConfig, tokenizer: PreTrainedTokenizer,  pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(lambda example: example[config.text_column_name] != "")

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))

    # Add a pad token if the tokenizer doesn't have one
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding = False if config.input_shapes["batch_size"] == 1 else True

    def tokenize_function(examples):
        return tokenizer(
            examples[config.text_column_name],
            truncation=config.truncation,
            return_token_type_ids=False,
            padding=padding,
            max_length = getattr(pretrained_config, "max_position_embeddings", 512)
            )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        writer_batch_size=50,
        remove_columns=dataset.features,
        desc="Running tokenizer on dataset",
    ).with_format("torch")

    return dataset


def image_preprocessing(dataset: Dataset, config: EnergyStarConfig, processor: PretrainedProcessor,  pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(lambda example: example[config.image_column_name] != "")

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))
        # Add a pad token if the tokenizer doesn't have one

    def preprocess_function(examples: Dict[str, Image]):
        processed = processor(examples[config.image_column_name].convert("RGB"))
        return {"pixel_values": processed["pixel_values"][0]}

    dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.features,
        desc="Running processor on dataset",
        writer_batch_size=50,
        batched=False,
    ).with_format("torch")

    return dataset


def image_to_text_preprocessing(dataset: Dataset, config: EnergyStarConfig, processor: PretrainedProcessor,  pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(lambda example: example[config.image_column_name] != "")

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))
        # Add a pad token if the tokenizer doesn't have one
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    def preprocess_function(examples):
        return processor(examples[config.image_column_name])

    dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.features,
        desc="Running processor on dataset",
        batched=True,
    ).with_format("torch")

    return dataset


def automatic_speech_recognition_preprocessing(
    dataset: Dataset, config: EnergyStarConfig, processor: PretrainedProcessor,  pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(lambda example: example[config.audio_column_name] != "")

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))
        # Add a pad token if the tokenizer doesn't have one
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    def preprocess_function(examples):
        audio = examples[config.audio_column_name]
        outputs = processor(audio["array"], sampling_rate=audio["sampling_rate"])

        # The processor may add an extra dimension so we squeeze it
        for key, value in outputs.items():
            if isinstance(value, list) and len(value) == 1:
                outputs[key] = value[0]
            elif isinstance(value, np.ndarray) and value.shape[0] == 1:
                outputs[key] = value.squeeze(0)

        return outputs

    dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.features,
        desc="Running processor on dataset",
    ).with_format("torch")

    return dataset


def sentence_similarity_preprocessing(
    dataset: Dataset, config: EnergyStarConfig, tokenizer: PreTrainedTokenizer,  pretrained_config: PretrainedConfig,
) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(
            lambda example: (example[config.sentence1_column_name], example[config.sentence2_column_name]) != ""
        )

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))

    # Add a pad token if the tokenizer doesn't have one
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding = False if config.input_shapes["batch_size"] == 1 else True

    def tokenize_function(examples):
        return tokenizer(
            examples[config.sentence1_column_name],
            examples[config.sentence2_column_name],
            padding=padding,
            truncation=config.truncation,
            max_length = getattr(pretrained_config, "max_position_embeddings", 512)
            )
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.features,
        desc="Running tokenizer on dataset",
    ).with_format("torch")

    return dataset
