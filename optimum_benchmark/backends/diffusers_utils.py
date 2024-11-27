import warnings
from typing import Dict

from hydra.utils import get_class

from ..import_utils import is_diffusers_available

if is_diffusers_available():
    import diffusers
    from diffusers import DiffusionPipeline


def get_diffusers_auto_pipeline_class_for_task(task: str):
    from ..task_utils import TASKS_TO_AUTO_PIPELINE_CLASS_NAMES

    if not is_diffusers_available():
        raise ImportError("diffusers is not available. Please, pip install diffusers.")

    model_loader_name = TASKS_TO_AUTO_PIPELINE_CLASS_NAMES.get(task, None)
    model_loader_class = getattr(diffusers, model_loader_name)
    return model_loader_class


def get_diffusers_pretrained_config(model: str, **kwargs) -> Dict[str, int]:
    if not is_diffusers_available():
        raise ImportError("diffusers is not available. Please, pip install diffusers.")

    config = DiffusionPipeline.load_config(model, **kwargs)
    pipeline_config = config[0] if isinstance(config, tuple) else config
    return pipeline_config


def extract_diffusers_shapes_from_model(model: str, **kwargs) -> Dict[str, int]:
    if not is_diffusers_available():
        raise ImportError("diffusers is not available. Please, pip install diffusers.")

    model_config = get_diffusers_pretrained_config(model, **kwargs)

    shapes = {}
    if "vae" in model_config:
        vae_import_path = model_config["vae"]
        vae_class = get_class(f"{vae_import_path[0]}.{vae_import_path[1]}")
        vae_config = vae_class.load_config(model, subfolder="vae", **kwargs)
        shapes["num_channels"] = vae_config["out_channels"]
        shapes["height"] = vae_config["sample_size"]
        shapes["width"] = vae_config["sample_size"]

    elif "vae_decoder" in model_config:
        vae_import_path = model_config["vae_decoder"]
        vae_class = get_class(f"{vae_import_path[0]}.{vae_import_path[1]}")
        vae_config = vae_class.load_config(model, subfolder="vae_decoder", **kwargs)
        shapes["num_channels"] = vae_config["out_channels"]
        shapes["height"] = vae_config["sample_size"]
        shapes["width"] = vae_config["sample_size"]

    elif "vae_encoder" in model_config:
        vae_import_path = model_config["vae_encoder"]
        vae_class = get_class(f"{vae_import_path[0]}.{vae_import_path[1]}")
        vae_config = vae_class.load_config(model, subfolder="vae_encoder", **kwargs)
        shapes["num_channels"] = vae_config["out_channels"]
        shapes["height"] = vae_config["sample_size"]
        shapes["width"] = vae_config["sample_size"]

    else:
        warnings.warn("Could not extract shapes [num_channels, height, width] from diffusion pipeline.")
        shapes["num_channels"] = -1
        shapes["height"] = -1
        shapes["width"] = -1

    return shapes
