from typing import Any, Dict

from transformers import PretrainedConfig

from ..import_utils import is_timm_available

if is_timm_available():
    import timm  # type: ignore


def get_timm_pretrained_config(model_name: str) -> PretrainedConfig:
    model_source, model_name = timm.models.parse_model_name(model_name)
    if model_source == "hf-hub":
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = timm.models.load_model_config_from_hf(model_name)
        return pretrained_cfg

    return timm.get_pretrained_cfg(model_name)


def extract_timm_shapes_from_config(config: PretrainedConfig) -> Dict[str, Any]:
    artifacts_dict = {}

    config_dict = {k: v for k, v in config.to_dict().items() if v is not None}
    artifacts_dict.update(config_dict)

    shapes = {}

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

    return shapes
