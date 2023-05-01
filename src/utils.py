import torch
from typing import Dict
from benchmark.config import BenchmarkConfig

from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification, \
    AutoModelForAudioClassification

from optimum.onnxruntime import ORTModelForSequenceClassification, \
    ORTModelForAudioClassification

TASK_TO_AUTOMODEL = {
    "sequence-classification": AutoModelForSequenceClassification,
    "audio-classification": AutoModelForAudioClassification
}

TASK_TO_ORTMODEL = {
    "sequence-classification": ORTModelForSequenceClassification,
    "audio-classification": ORTModelForAudioClassification
}


def get_input_ids(config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
    return torch.randint(
        low=0,
        high=AutoConfig.from_pretrained(config.model).vocab_size,
        size=(config.batch_size, config.sequence_length),
        dtype=torch.long,
        device=config.backend.device,
    )


def get_attention_mask(config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
    return torch.ones(
        config.batch_size,
        config.sequence_length,
        dtype=torch.long,
        device=config.backend.device,
    )


def get_token_ids(config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
    return torch.ones(
        config.batch_size,
        config.sequence_length,
        dtype=torch.long,
        device=config.backend.device,
    )


INPUT_GENERATORS = {
    'input_ids': get_input_ids,
    'attention_mask': get_attention_mask,
    'token_type_ids': get_token_ids
}