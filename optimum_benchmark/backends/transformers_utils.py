from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    FeatureExtractionMixin,
    GenerationConfig,
    ImageProcessingMixin,
    PretrainedConfig,
    PreTrainedTokenizer,
    ProcessorMixin,
)

PretrainedProcessor = Union[FeatureExtractionMixin, ImageProcessingMixin, PreTrainedTokenizer, ProcessorMixin]


def get_transformers_pretrained_config(model: str, **kwargs) -> "PretrainedConfig":
    # sometimes contains information about the model's input shapes that are not available in the config
    return AutoConfig.from_pretrained(model, **kwargs)


def get_transformers_generation_config(model: str, **kwargs) -> Optional["GenerationConfig"]:
    try:
        # sometimes contains information about the model's input shapes that are not available in the config
        return GenerationConfig.from_pretrained(model, **kwargs)
    except Exception:
        return GenerationConfig()


def get_transformers_pretrained_processor(model: str, **kwargs) -> Optional["PretrainedProcessor"]:
    try:
        # sometimes contains information about the model's input shapes that are not available in the config
        return AutoProcessor.from_pretrained(model, **kwargs)
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(model, **kwargs)
        except Exception:
            return None


def extract_transformers_shapes_from_artifacts(
    config: "PretrainedConfig", processor: Optional["PretrainedProcessor"] = None
) -> Dict[str, Any]:
    artifacts_dict = {}

    config_dict = {k: v for k, v in config.to_dict().items() if v is not None}
    artifacts_dict.update(config_dict)

    if processor is not None and hasattr(processor, "to_dict"):
        processor_dict = {k: v for k, v in processor.to_dict().items() if v is not None}
        artifacts_dict.update(processor_dict)
    elif processor is not None:
        processor_dict = {k: getattr(processor, k) for k in dir(processor) if isinstance(getattr(processor, k), int)}

    shapes = {}

    # text input
    shapes["vocab_size"] = artifacts_dict.get("vocab_size", None)
    shapes["type_vocab_size"] = artifacts_dict.get("type_vocab_size", None)
    shapes["max_position_embeddings"] = artifacts_dict.get("max_position_embeddings", None)
    if shapes["max_position_embeddings"] is None:
        shapes["max_position_embeddings"] = artifacts_dict.get("n_positions", None)

    # image input
    shapes["num_channels"] = artifacts_dict.get("num_channels", None)
    if shapes["num_channels"] is None:
        # processors have different names for the number of channels
        shapes["num_channels"] = artifacts_dict.get("channels", None)

    image_size = artifacts_dict.get("image_size", None)
    if image_size is None:
        # processors have different names for the image size
        image_size = artifacts_dict.get("size", None)

    if isinstance(image_size, (int, float)):
        shapes["height"] = image_size
        shapes["width"] = image_size
    elif isinstance(image_size, (list, tuple)):
        shapes["height"] = image_size[0]
        shapes["width"] = image_size[0]
    elif isinstance(image_size, dict) and len(image_size) == 2:
        shapes["height"] = list(image_size.values())[0]
        shapes["width"] = list(image_size.values())[1]
    elif isinstance(image_size, dict) and len(image_size) == 1:
        shapes["height"] = list(image_size.values())[0]
        shapes["width"] = list(image_size.values())[0]
    else:
        shapes["height"] = None
        shapes["width"] = None

    input_size = artifacts_dict.get("input_size", None)
    if input_size is not None:
        shapes["num_channels"] = input_size[0]
        shapes["height"] = input_size[1]
        shapes["width"] = input_size[2]

    # classification labels
    id2label = artifacts_dict.get("id2label", None)
    if id2label is not None:
        shapes["num_labels"] = len(id2label)

    num_classes = artifacts_dict.get("num_classes", None)
    if num_classes is not None:
        shapes["num_labels"] = num_classes

    # object detection labels
    shapes["num_queries"] = artifacts_dict.get("num_queries", None)
    if shapes["num_queries"] == 0:
        shapes["num_queries"] = 2

    return shapes


TORCH_INIT_FUNCTIONS = {
    "normal_": torch.nn.init.normal_,
    "uniform_": torch.nn.init.uniform_,
    "trunc_normal_": torch.nn.init.trunc_normal_,
    "xavier_normal_": torch.nn.init.xavier_normal_,
    "xavier_uniform_": torch.nn.init.xavier_uniform_,
    "kaiming_normal_": torch.nn.init.kaiming_normal_,
    "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
    "normal": torch.nn.init.normal,
    "uniform": torch.nn.init.uniform,
    "xavier_normal": torch.nn.init.xavier_normal,
    "xavier_uniform": torch.nn.init.xavier_uniform,
    "kaiming_normal": torch.nn.init.kaiming_normal,
    "kaiming_uniform": torch.nn.init.kaiming_uniform,
}


def fast_rand(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.nn.init.uniform_(tensor)


@contextmanager
def random_init_weights():
    # Replace the initialization functions
    for name, init_func in TORCH_INIT_FUNCTIONS.items():
        if name != "uniform_":
            setattr(torch.nn.init, name, fast_rand)
    try:
        yield
    finally:
        # Restore the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            if name != "uniform_":
                setattr(torch.nn.init, name, init_func)
