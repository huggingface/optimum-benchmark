from typing import Dict

from hydra.utils import get_class

from ..import_utils import is_diffusers_available
from ..task_utils import _DIFFUSERS_TASKS_TO_MODEL_LOADERS, map_from_synonym

if is_diffusers_available():
    import diffusers
    from diffusers import DiffusionPipeline


def get_diffusers_pretrained_config(model: str, **kwargs) -> Dict[str, int]:
    return DiffusionPipeline.load_config(model, **kwargs)


def extract_diffusers_shapes_from_model(model: str, **kwargs) -> Dict[str, int]:
    model_config = get_diffusers_pretrained_config(model, **kwargs)

    shapes = {}
    if "vae" in model_config:
        vae_import_path = model_config["vae"]
        vae_class = get_class(f"{vae_import_path[0]}.{vae_import_path[1]}")
        vae_config = vae_class.load_config(model, subfolder="vae", **kwargs)
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
        shapes["num_channels"] = -1
        shapes["height"] = -1
        shapes["width"] = -1

    return shapes


def get_diffusers_automodel_loader_for_task(task: str):
    task = map_from_synonym(task)
    model_loader_name = _DIFFUSERS_TASKS_TO_MODEL_LOADERS[task]
    model_loader_class = getattr(diffusers, model_loader_name)
    return model_loader_class
