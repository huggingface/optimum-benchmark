from typing import Any, Dict

from transformers import PretrainedConfig

from ..import_utils import is_timm_available

if is_timm_available():
    from timm import create_model
    from timm.models import get_pretrained_cfg, load_model_config_from_hf, parse_model_name


def get_timm_model_creator():
    if not is_timm_available():
        raise ImportError("timm is not available. Please, pip install timm.")

    return create_model


def get_timm_pretrained_config(model_name: str) -> "PretrainedConfig":
    if not is_timm_available():
        raise ImportError("timm is not available. Please, pip install timm.")

    model_source, model_name = parse_model_name(model_name)
    if model_source == "hf-hub":
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)
        return pretrained_cfg

    return get_pretrained_cfg(model_name)


def extract_timm_shapes_from_config(config: "PretrainedConfig") -> Dict[str, Any]:
    if not is_timm_available():
        raise ImportError("timm is not available. Please, pip install timm.")

    artifacts_dict = {}

    config_dict = {k: v for k, v in config.to_dict().items() if v is not None}
    artifacts_dict.update(config_dict)

    shapes = {}

    # image input
    if "num_channels" in artifacts_dict:
        shapes["num_channels"] = artifacts_dict.get("num_channels", None)
    elif "channels" in artifacts_dict:
        shapes["num_channels"] = artifacts_dict.get("channels", None)

    if "image_size" in artifacts_dict:
        image_size = artifacts_dict["image_size"]
    elif "size" in artifacts_dict:
        image_size = artifacts_dict["size"]
    else:
        image_size = None

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

    if "input_size" in artifacts_dict:
        input_size = artifacts_dict.get("input_size", None)
        shapes["num_channels"] = input_size[0]
        shapes["height"] = input_size[1]
        shapes["width"] = input_size[2]

    return shapes
