from typing import Dict, List

import numpy as np
from datasets import Dataset
from PIL.Image import Image
from transformers import PretrainedConfig

from ..backends.transformers_utils import PretrainedProcessor
from ..scenarios import EnergyStarConfig


def feature_extraction_preprocessing(
    dataset: Dataset,
    pretrained_processor: PretrainedProcessor,
    scenario_config: EnergyStarConfig,
    pretrained_config: PretrainedConfig,
) -> Dataset:
    if scenario_config.input_shapes["batch_size"] == 1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.filter(lambda example: example[scenario_config.text_column_name] != "")

    if scenario_config.num_samples != -1:
        dataset = dataset.select(range(scenario_config.num_samples))

    if getattr(pretrained_processor, "pad_token", None) is None:
        pretrained_processor.pad_token = pretrained_processor.eos_token

    padding = scenario_config.input_shapes["batch_size"] != 1
    max_length = getattr(pretrained_config, "max_position_embeddings", 512)

    def tokenize_function(examples):
        return pretrained_processor(
            examples[scenario_config.text_column_name],
            truncation=scenario_config.truncation,
            max_length=max_length,
            padding=padding,
        )

    dataset = dataset.map(
        function=tokenize_function,
        desc="Running tokenizer on dataset",
        remove_columns=dataset.features,
        writer_batch_size=50,
        batched=True,
    ).with_format("torch")

    return dataset


def summarization_preprocessing(
    dataset: Dataset,
    pretrained_processor: PretrainedProcessor,
    scenario_config: EnergyStarConfig,
    pretrained_config: PretrainedConfig,
) -> Dataset:
    if scenario_config.input_shapes["batch_size"] == 1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.filter(lambda example: example[scenario_config.text_column_name] != "")

    if scenario_config.num_samples != -1:
        dataset = dataset.select(range(scenario_config.num_samples))

    if getattr(pretrained_processor, "pad_token", None) is None:
        pretrained_processor.pad_token = pretrained_processor.eos_token

    padding = scenario_config.input_shapes["batch_size"] != 1
    max_length = getattr(pretrained_config, "max_position_embeddings", 512)

    def tokenize_function(examples):
        return pretrained_processor(
            examples[scenario_config.text_column_name],
            truncation=scenario_config.truncation,
            max_length=max_length,
            padding=padding,
        )

    dataset = dataset.map(
        function=tokenize_function,
        desc="Running tokenizer on dataset",
        remove_columns=dataset.features,
        writer_batch_size=50,
        batched=True,
    ).with_format("torch")

    return dataset


def text_classification_preprocessing(
    dataset: Dataset,
    pretrained_processor: PretrainedProcessor,
    scenario_config: EnergyStarConfig,
    pretrained_config: PretrainedConfig,
) -> Dataset:
    if scenario_config.input_shapes["batch_size"] == 1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.filter(lambda example: example[scenario_config.text_column_name] != "")

    if scenario_config.num_samples != -1:
        dataset = dataset.select(range(scenario_config.num_samples))

    if getattr(pretrained_processor, "pad_token", None) is None:
        pretrained_processor.pad_token = pretrained_processor.eos_token

    padding = scenario_config.input_shapes["batch_size"] != 1
    max_length = getattr(pretrained_config, "max_position_embeddings", 512)

    def tokenize_function(examples):
        return pretrained_processor(
            examples[scenario_config.text_column_name],
            truncation=scenario_config.truncation,
            max_length=max_length,
            padding=padding,
        )

    dataset = dataset.map(
        function=tokenize_function,
        desc="Running tokenizer on dataset",
        remove_columns=dataset.features,
        writer_batch_size=50,
        batched=True,
    ).with_format("torch")

    return dataset


def question_answering_preprocessing(
    dataset: Dataset,
    pretrained_processor: PretrainedProcessor,
    scenario_config: EnergyStarConfig,
    pretrained_config: PretrainedConfig,
) -> Dataset:
    if scenario_config.input_shapes["batch_size"] == 1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.filter(
            lambda example: (
                example[scenario_config.question_column_name] != ""
                and example[scenario_config.context_column_name] != ""
            )
        )

    if scenario_config.num_samples != -1:
        dataset = dataset.select(range(scenario_config.num_samples))

    if getattr(pretrained_processor, "pad_token", None) is None:
        pretrained_processor.pad_token = pretrained_processor.eos_token

    padding = scenario_config.input_shapes["batch_size"] != 1
    max_length = getattr(pretrained_config, "max_position_embeddings", 512)

    def tokenize_function(examples):
        return pretrained_processor(
            examples[scenario_config.question_column_name],
            examples[scenario_config.context_column_name],
            truncation=scenario_config.truncation,
            max_length=max_length,
            padding=padding,
        )

    dataset = dataset.map(
        function=tokenize_function,
        desc="Running tokenizer on dataset",
        remove_columns=dataset.features,
        writer_batch_size=50,
        batched=True,
    ).with_format("torch")

    return dataset


def text2text_generation_preprocessing(
    dataset: Dataset,
    pretrained_processor: PretrainedProcessor,
    scenario_config: EnergyStarConfig,
    pretrained_config: PretrainedConfig,
) -> Dataset:
    if scenario_config.num_samples != -1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.select(range(scenario_config.num_samples))

    if getattr(pretrained_processor, "pad_token", None) is None:
        pretrained_processor.pad_token = pretrained_processor.eos_token

    padding = scenario_config.input_shapes["batch_size"] != 1
    max_length = getattr(pretrained_config, "max_position_embeddings", 512)
    len_prefix1 = len(pretrained_processor(scenario_config.dataset_prefix1))
    len_prefix2 = len(pretrained_processor(scenario_config.dataset_prefix2))
    new_tokens = scenario_config.generate_kwargs["max_new_tokens"]

    def add_single_prefix(example):
        example[scenario_config.text_column_name] = (
            scenario_config.dataset_prefix1 + example[scenario_config.text_column_name]
        )
        return example

    def add_qa_prefix(example):
        example[scenario_config.text_column_name] = (
            scenario_config.dataset_prefix1
            + example[scenario_config.question_column_name]
            + scenario_config.dataset_prefix2
            + example[scenario_config.context_column_name]
        )
        return example

    def tokenize_function_qa(examples):
        return pretrained_processor(
            examples[scenario_config.text_column_name],
            max_length=max_length - len_prefix1 - len_prefix2,
            truncation=scenario_config.truncation,
            return_token_type_ids=False,
            padding=padding,
        )

    def tokenize_function_single(examples):
        return pretrained_processor(
            examples[scenario_config.text_column_name],
            truncation=scenario_config.truncation,
            max_length=max_length - len_prefix1,
            return_token_type_ids=False,
            padding=padding,
        )

    def tokenize_function_generation(examples):
        return pretrained_processor(
            examples[scenario_config.text_column_name],
            truncation=scenario_config.truncation,
            max_length=max_length - new_tokens,
            return_token_type_ids=False,
            padding=padding,
        )

    if scenario_config.t5_task in ["question_answering"]:
        dataset = dataset.map(add_qa_prefix)
        dataset = dataset.map(
            function=tokenize_function_qa,
            desc="Running tokenizer on dataset",
            remove_columns=dataset.features,
            writer_batch_size=50,
            batched=True,
        ).with_format("torch")

    elif scenario_config.t5_task in ["text_generation"]:
        dataset = dataset.map(
            function=tokenize_function_generation,
            desc="Running tokenizer on dataset",
            remove_columns=dataset.features,
            writer_batch_size=50,
            batched=True,
        ).with_format("torch")

    elif scenario_config.t5_task in ["text_classification", "summarization"]:
        dataset = dataset.map(add_single_prefix)
        dataset = dataset.map(
            function=tokenize_function_single,
            desc="Running tokenizer on dataset",
            remove_columns=dataset.features,
            writer_batch_size=50,
            batched=True,
        ).with_format("torch")

    else:
        raise ValueError(f"T5 task {scenario_config.t5_task} not supported for text2text-generation")

    return dataset


def text_generation_preprocessing(
    dataset: Dataset,
    pretrained_processor: PretrainedProcessor,
    scenario_config: EnergyStarConfig,
    pretrained_config: PretrainedConfig,
) -> Dataset:
    if scenario_config.input_shapes["batch_size"] == 1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.filter(lambda example: example[scenario_config.text_column_name] != "")

    if scenario_config.num_samples != -1:
        dataset = dataset.select(range(scenario_config.num_samples))

    if getattr(pretrained_processor, "pad_token", None) is None:
        pretrained_processor.pad_token = pretrained_processor.eos_token

    padding = scenario_config.input_shapes["batch_size"] != 1
    max_length = getattr(pretrained_config, "max_position_embeddings", 512)
    new_tokens = scenario_config.generate_kwargs["max_new_tokens"]

    def tokenize_function(examples):
        return pretrained_processor(
            examples[scenario_config.text_column_name],
            truncation=scenario_config.truncation,
            max_length=max_length - new_tokens,
            return_token_type_ids=False,
            padding=padding,
        )

    def reasoning_tokenize_function(examples):
        return pretrained_processor.apply_chat_template(
            [{"role": "user", "content": examples[scenario_config.text_column_name]}],
            truncation=scenario_config.truncation,
            max_length=min(max_length, 2048) - new_tokens,
            padding=padding,
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=True,
            return_dict=True,
            **scenario_config.reasoning_params,
        )

    if scenario_config.reasoning:
        dataset = dataset.map(
            function=reasoning_tokenize_function,
            desc="Running reasoning tokenizer on dataset",
            remove_columns=dataset.features,
        ).with_format("torch")

    else:
        dataset = dataset.map(
            function=tokenize_function,
            desc="Running tokenizer on dataset",
            remove_columns=dataset.features,
            writer_batch_size=50,
            batched=True,
        ).with_format("torch")

    return dataset


def image_preprocessing(
    dataset: Dataset,
    pretrained_processor: PretrainedProcessor,
    scenario_config: EnergyStarConfig,
    **kwargs,
) -> Dataset:
    if scenario_config.input_shapes["batch_size"] == 1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.filter(lambda example: example[scenario_config.image_column_name] != "")

    if scenario_config.num_samples != -1:
        dataset = dataset.select(range(scenario_config.num_samples))

    def preprocess_function(examples: Dict[str, List[Image]]):
        return pretrained_processor([image.convert("RGB") for image in examples[scenario_config.image_column_name]])

    dataset = dataset.map(
        function=preprocess_function,
        desc="Running processor on dataset",
        remove_columns=dataset.features,
        writer_batch_size=50,
        batched=True,
    ).with_format("torch")

    return dataset


def image_to_text_preprocessing(
    dataset: Dataset,
    pretrained_processor: PretrainedProcessor,
    scenario_config: EnergyStarConfig,
    **kwargs,
) -> Dataset:
    if scenario_config.input_shapes["batch_size"] == 1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.filter(lambda example: example[scenario_config.image_column_name] != "")

    if scenario_config.num_samples != -1:
        dataset = dataset.select(range(scenario_config.num_samples))

    if getattr(pretrained_processor.tokenizer, "pad_token", None) is None:
        pretrained_processor.tokenizer.pad_token = pretrained_processor.tokenizer.eos_token

    def preprocess_function(examples):
        return pretrained_processor(images=examples[scenario_config.image_column_name])

    dataset = dataset.map(
        function=preprocess_function,
        desc="Running processor on dataset",
        remove_columns=dataset.features,
        writer_batch_size=50,
        batched=True,
    ).with_format("torch")

    return dataset


def automatic_speech_recognition_preprocessing(
    dataset: Dataset,
    pretrained_processor: PretrainedProcessor,
    scenario_config: EnergyStarConfig,
    pretrained_config: PretrainedConfig,
) -> Dataset:
    if scenario_config.input_shapes["batch_size"] == 1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.filter(lambda example: example[scenario_config.audio_column_name] != "")

    if scenario_config.num_samples != -1:
        dataset = dataset.select(range(scenario_config.num_samples))

        pretrained_processor.tokenizer.pad_token = pretrained_processor.tokenizer.eos_token

    def preprocess_function(examples: Dict[str, Dict[str, np.ndarray]]):
        audios = [audio["array"] for audio in examples[scenario_config.audio_column_name]]
        sampling_rates = examples[scenario_config.audio_column_name][0]["sampling_rate"]
        outputs = pretrained_processor(audios, sampling_rate=sampling_rates)
        return outputs

    dataset = dataset.map(
        preprocess_function,
        desc="Running processor on dataset",
        remove_columns=dataset.features,
        writer_batch_size=50,
        batched=True,
    ).with_format("torch")

    return dataset


def sentence_similarity_preprocessing(
    dataset: Dataset,
    pretrained_processor: PretrainedProcessor,
    scenario_config: EnergyStarConfig,
    pretrained_config: PretrainedConfig,
) -> Dataset:
    if scenario_config.input_shapes["batch_size"] == 1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.filter(
            lambda example: (example[scenario_config.sentence1_column_name] != "")
            and (example[scenario_config.sentence2_column_name] != "")
        )

    if scenario_config.num_samples != -1:
        dataset = dataset.select(range(scenario_config.num_samples))

    if getattr(pretrained_processor, "pad_token", None) is None:
        pretrained_processor.pad_token = pretrained_processor.eos_token

    padding = scenario_config.input_shapes["batch_size"] != 1
    max_length = getattr(pretrained_config, "max_position_embeddings", 512)

    def tokenize_function(examples):
        return pretrained_processor(
            examples[scenario_config.sentence1_column_name],
            examples[scenario_config.sentence2_column_name],
            truncation=scenario_config.truncation,
            max_length=max_length,
            padding=padding,
        )

    dataset = dataset.map(
        function=tokenize_function,
        desc="Running tokenizer on dataset",
        remove_columns=dataset.features,
        writer_batch_size=50,
        batched=True,
    ).with_format("torch")

    return dataset


def text_to_image_preprocessing(
    dataset: Dataset,
    scenario_config: EnergyStarConfig,
    **kwargs,
) -> Dataset:
    if scenario_config.input_shapes["batch_size"] == 1:
        # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
        dataset = dataset.filter(lambda example: example[scenario_config.text_column_name] != "")

    if scenario_config.num_samples != -1:
        dataset = dataset.select(range(scenario_config.num_samples))

    return dataset


TASKS_TO_PREPROCESSORS = {
    "automatic-speech-recognition": automatic_speech_recognition_preprocessing,
    "text2text-generation": text2text_generation_preprocessing,
    "sentence-similarity": sentence_similarity_preprocessing,
    "text-classification": text_classification_preprocessing,
    "question-answering": question_answering_preprocessing,
    "feature-extraction": feature_extraction_preprocessing,
    "text-generation": text_generation_preprocessing,
    "summarization": summarization_preprocessing,
    "text-to-image": text_to_image_preprocessing,
    "image-to-text": image_to_text_preprocessing,
    "image-classification": image_preprocessing,
    "object-detection": image_preprocessing,
}
