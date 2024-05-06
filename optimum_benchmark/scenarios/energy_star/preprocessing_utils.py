from datasets import Dataset
from transformers import PreTrainedTokenizer

from ...backends.transformers_utils import PretrainedProcessor
from .config import EnergyStarConfig


def preprocess(dataset: Dataset, task: str, config: EnergyStarConfig, preprocessor: PretrainedProcessor) -> Dataset:
    task_to_preprocessing = {"feature-extraction": feature_extraction_preprocessing}

    return task_to_preprocessing[task](dataset, config, preprocessor)


def feature_extraction_preprocessing(
    dataset: Dataset, config: EnergyStarConfig, tokenizer: PreTrainedTokenizer
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
            max_length=config.max_length if config.max_length != -1 else None,
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
