from typing import Dict

from ..import_utils import is_diffusers_available
from ..task_utils import TASKS_TO_AUTO_PIPELINE_CLASS_NAMES, map_from_synonym_task

if is_diffusers_available():
    import diffusers
    from diffusers import DiffusionPipeline


def get_diffusers_auto_pipeline_class_for_task(task: str):
    task = map_from_synonym_task(task)

    if not is_diffusers_available():
        raise ImportError("diffusers is not available. Please, pip install diffusers.")

    if task not in TASKS_TO_AUTO_PIPELINE_CLASS_NAMES:
        raise ValueError(f"Task {task} not supported for diffusers")

    model_loader_name = TASKS_TO_AUTO_PIPELINE_CLASS_NAMES[task]

    return getattr(diffusers, model_loader_name)


def get_diffusers_pretrained_config(model: str, **kwargs) -> Dict[str, int]:
    if not is_diffusers_available():
        raise ImportError("diffusers is not available. Please, pip install diffusers.")

    config = DiffusionPipeline.load_config(model, **kwargs)
    pipeline_config = config[0] if isinstance(config, tuple) else config
    return pipeline_config


def extract_diffusers_shapes_from_model(**kwargs) -> Dict[str, int]:
    if not is_diffusers_available():
        raise ImportError("diffusers is not available. Please, pip install diffusers.")

    shapes = {}

    return shapes
