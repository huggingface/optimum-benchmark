from typing import Any, Dict, Optional, Union

from transformers import (
    FeatureExtractionMixin,
    ImageProcessingMixin,
    Pipeline,
    PretrainedConfig,
    PreTrainedTokenizer,
    ProcessorMixin,
)

PreTrainedProcessor = Union[
    PreTrainedTokenizer,
    ImageProcessingMixin,
    FeatureExtractionMixin,
    ProcessorMixin,
]


def extract_shapes_from_diffusion_pipeline(pipeline: Pipeline) -> Dict[str, Any]:
    # this is the only way I found to extract a diffusion pipeline's "input" shapes
    shapes = {}
    if hasattr(pipeline, "vae_encoder") and hasattr(pipeline.vae_encoder, "config"):
        shapes["num_channels"] = pipeline.vae_encoder.config["out_channels"]
        shapes["height"] = pipeline.vae_encoder.config["sample_size"]
        shapes["width"] = pipeline.vae_encoder.config["sample_size"]
    elif hasattr(pipeline, "vae") and hasattr(pipeline.vae, "config"):
        shapes["num_channels"] = pipeline.vae.config.out_channels
        shapes["height"] = pipeline.vae.config.sample_size
        shapes["width"] = pipeline.vae.config.sample_size
    else:
        shapes["num_channels"] = -1
        shapes["height"] = -1
        shapes["width"] = -1

    return shapes


def extract_shapes_from_model_artifacts(
    config: "PretrainedConfig",
    processor: Optional["PreTrainedProcessor"] = None,
) -> Dict[str, Any]:
    shapes = {}
    artifacts_dict = {}

    config_dict = {k: v for k, v in config.to_dict().items() if v is not None}
    artifacts_dict.update(config_dict)

    if processor is not None and hasattr(processor, "to_dict"):
        processor_dict = {k: v for k, v in processor.to_dict().items() if v is not None}
        artifacts_dict.update(processor_dict)

    # text input
    shapes["vocab_size"] = artifacts_dict.get("vocab_size", None)
    if shapes["vocab_size"] is None or shapes["vocab_size"] == 0:
        shapes["vocab_size"] = 2

    shapes["type_vocab_size"] = artifacts_dict.get("type_vocab_size", None)
    if shapes["type_vocab_size"] is None or shapes["type_vocab_size"] == 0:
        shapes["type_vocab_size"] = 2

    shapes["max_position_embeddings"] = artifacts_dict.get("max_position_embeddings", None)
    if shapes["max_position_embeddings"] is None or shapes["max_position_embeddings"] == 0:
        shapes["max_position_embeddings"] = 2

    # image input
    shapes["num_channels"] = artifacts_dict.get("num_channels", None)
    if shapes["num_channels"] is None or shapes["num_channels"] == 0:
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

    # classification labels
    id2label = artifacts_dict.get("id2label", {"0": "LABEL_0", "1": "LABEL_1"})
    shapes["num_labels"] = len(id2label)
    if shapes["num_labels"] == 0:
        shapes["num_labels"] = 2

    # object detection labels
    shapes["num_queries"] = artifacts_dict.get("num_queries", None)
    if shapes["num_queries"] == 0:
        shapes["num_queries"] = 2

    return shapes
