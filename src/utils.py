import torch
from typing import Dict
from transformers import AutoConfig

from src.benchmark.config import BenchmarkConfig


def get_input_ids(config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
    """Generate random input ids"""
    return torch.randint(
        low=0,
        high=AutoConfig.from_pretrained(config.model).vocab_size,
        size=(config.batch_size, config.sequence_length),
        dtype=torch.long,
        device=config.backend.device,
    )


def get_attention_mask(config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
    """Generate random attention mask with config.sparsity ratio of upper triangular values set to 0"""
    attention_mask = torch.ones(
        config.batch_size,
        config.sequence_length,
        dtype=torch.long,
        device=config.backend.device,
    )

    if config.sparsity > 0:
        # apply sparse mask
        mask = torch.rand(
            (config.batch_size, config.sequence_length), device=config.backend.device)
        attention_mask[mask < config.sparsity] = 0
        attention_mask, _ = attention_mask.sort(dim=-1, descending=True)

    return attention_mask


def get_token_ids(config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
    """Generate random token type ids"""
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
