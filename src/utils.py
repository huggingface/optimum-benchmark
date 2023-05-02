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
    """Generate random attention mask"""
    mask = torch.ones(
        config.batch_size,
        config.sequence_length,
        dtype=torch.long,
        device=config.backend.device,
    )
    
    # masking out a certain ratio (config.sparsity) of tokens
    mask = mask * torch.distributions.Bernoulli(
        torch.tensor([config.sparsity], device=config.backend.device)
    ).sample((config.batch_size, config.sequence_length)).long()

    return mask


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