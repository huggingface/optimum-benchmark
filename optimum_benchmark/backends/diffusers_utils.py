import warnings
from typing import Dict

from hydra.utils import get_class

from ..import_utils import is_diffusers_available

if is_diffusers_available():
    import diffusers
    from diffusers import DiffusionPipeline

    if hasattr(diffusers, "pipelines") and hasattr(diffusers.pipelines, "auto_pipeline"):
        from diffusers.pipelines.auto_pipeline import (
            AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
            AUTO_INPAINT_PIPELINES_MAPPING,
            AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
        )

        TASKS_TO_MODEL_TYPES_TO_MODEL_CLASSES = {
            "inpainting": AUTO_INPAINT_PIPELINES_MAPPING.copy(),
            "text-to-image": AUTO_TEXT2IMAGE_PIPELINES_MAPPING.copy(),
            "image-to-image": AUTO_IMAGE2IMAGE_PIPELINES_MAPPING.copy(),
        }

        for task_name, model_mapping in TASKS_TO_MODEL_TYPES_TO_MODEL_CLASSES.items():
            for model_type, model_class in model_mapping.items():
                TASKS_TO_MODEL_TYPES_TO_MODEL_CLASSES[task_name][model_type] = model_class.__name__
    else:
        TASKS_TO_MODEL_TYPES_TO_MODEL_CLASSES = {}
else:
    TASKS_TO_MODEL_TYPES_TO_MODEL_CLASSES = {}


TASKS_TO_MODEL_LOADERS = {
    "inpainting": "AutoPipelineForInpainting",
    "text-to-image": "AutoPipelineForText2Image",
    "image-to-image": "AutoPipelineForImage2Image",
}


def get_diffusers_pretrained_config(model: str, **kwargs) -> Dict[str, int]:
    config = DiffusionPipeline.load_config(model, **kwargs)
    pipeline_config = config[0] if isinstance(config, tuple) else config
    return pipeline_config


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
        warnings.warn("Could not extract shapes [num_channels, height, width] from diffusion pipeline.")
        shapes["num_channels"] = -1
        shapes["height"] = -1
        shapes["width"] = -1

    return shapes


def get_diffusers_automodel_loader_for_task(task: str):
    model_loader_name = TASKS_TO_MODEL_LOADERS[task]
    model_loader_class = getattr(diffusers, model_loader_name)
    return model_loader_class
