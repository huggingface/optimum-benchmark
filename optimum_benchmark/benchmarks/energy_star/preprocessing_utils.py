import torch
from datasets import Dataset
from transformers import AutoTokenizer

from .config import EnergyStarConfig


def preprocess(dataset: Dataset, task: str, config: EnergyStarConfig, model_name: str) -> Dataset:
    task_to_preprocessing = {"feature-extraction": feature_extraction_preprocessing}

    return task_to_preprocessing[task](dataset, config, model_name)


def feature_extraction_preprocessing(dataset: Dataset, config: EnergyStarConfig, model_name: str) -> Dataset:
    # Remove empty samples when batch_size is 1 because empty inputs will make the model fail
    if config.input_shapes["batch_size"] == 1:
        dataset = dataset.filter(lambda example: example[config.text_column_name] != "")

    if config.num_samples != -1:
        dataset = dataset.select(range(config.num_samples))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add a pad token if the tokenizer doesn't have one
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding = False if config.input_shapes["batch_size"] == 1 else True

    def tokenize_function(examples):
        return tokenizer(examples[config.text_column_name], padding=padding)

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.features,
        desc="Running tokenizer on dataset",
    )

    # We don't want a torch dependency here but for now the only backend used for this benchmark is PyTorch
    def pt_transform(batch):
        return {key: torch.tensor(val) for key, val in batch.items()}

    dataset = dataset.with_transform(pt_transform)

    return dataset
