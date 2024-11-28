from contextlib import contextmanager
from typing import Any, Dict, Optional, Type, Union

import torch
import transformers
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    FeatureExtractionMixin,
    GenerationConfig,
    ImageProcessingMixin,
    PretrainedConfig,
    ProcessorMixin,
    SpecialTokensMixin,
)

from ..task_utils import TASKS_TO_AUTO_MODEL_CLASS_NAMES, map_from_synonym_task


def get_transformers_auto_model_class_for_task(task: str, model_type: Optional[str] = None) -> Type["AutoModel"]:
    task = map_from_synonym_task(task)

    if task not in TASKS_TO_AUTO_MODEL_CLASS_NAMES:
        raise ValueError(f"Task {task} not supported for transformers")

    if isinstance(TASKS_TO_AUTO_MODEL_CLASS_NAMES[task], str):
        return getattr(transformers, TASKS_TO_AUTO_MODEL_CLASS_NAMES[task])
    else:
        if model_type is None:
            raise ValueError(f"Task {task} requires a model_type to be specified")

        for automodel_class_name in TASKS_TO_AUTO_MODEL_CLASS_NAMES[task]:
            automodel_class = getattr(transformers, automodel_class_name)
            if model_type in automodel_class._model_mapping._model_mapping:
                return automodel_class

    raise ValueError(f"Task {task} not supported for model type {model_type}")


PretrainedProcessor = Union["FeatureExtractionMixin", "ImageProcessingMixin", "SpecialTokensMixin", "ProcessorMixin"]


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
            return AutoFeatureExtractor.from_pretrained(model, **kwargs)
        except Exception:
            try:
                return AutoImageProcessor.from_pretrained(model, **kwargs)
            except Exception:
                try:
                    return AutoTokenizer.from_pretrained(model, **kwargs)
                except Exception:
                    return None


def get_flat_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update(get_flat_dict(v))
        else:
            flat_dict[k] = v
    return flat_dict


def get_flat_artifact_dict(artifact: Union["PretrainedConfig", "PretrainedProcessor"]) -> Dict[str, Any]:
    artifact_dict = {}

    if isinstance(artifact, ProcessorMixin):
        artifact_dict.update(
            {k: v for k, v in artifact.__dict__.items() if isinstance(v, (int, str, float, bool, list, tuple, dict))}
        )
        for attribute in artifact.attributes:
            artifact_dict.update(get_flat_artifact_dict(getattr(artifact, attribute)))
    elif hasattr(artifact, "to_dict"):
        artifact_dict.update(
            {k: v for k, v in artifact.to_dict().items() if isinstance(v, (int, str, float, bool, list, tuple, dict))}
        )
    else:
        artifact_dict.update(
            {k: v for k, v in artifact.__dict__.items() if isinstance(v, (int, str, float, bool, list, tuple, dict))}
        )

    artifact_dict = get_flat_dict(artifact_dict)

    return artifact_dict


def extract_transformers_shapes_from_artifacts(
    config: Optional["PretrainedConfig"] = None,
    processor: Optional["PretrainedProcessor"] = None,
) -> Dict[str, Any]:
    flat_artifacts_dict = {}

    if config is not None:
        flat_artifacts_dict.update(get_flat_artifact_dict(config))

    if processor is not None:
        flat_artifacts_dict.update(get_flat_artifact_dict(processor))

    shapes = {}

    # text input
    if "vocab_size" in flat_artifacts_dict:
        shapes["vocab_size"] = flat_artifacts_dict["vocab_size"]

    if "type_vocab_size" in flat_artifacts_dict:
        shapes["type_vocab_size"] = flat_artifacts_dict["type_vocab_size"]

    if "max_position_embeddings" in flat_artifacts_dict:
        shapes["max_position_embeddings"] = flat_artifacts_dict["max_position_embeddings"]
    elif "n_positions" in flat_artifacts_dict:
        shapes["max_position_embeddings"] = flat_artifacts_dict["n_positions"]

    # image input
    if "num_channels" in flat_artifacts_dict:
        shapes["num_channels"] = flat_artifacts_dict["num_channels"]

    if "image_size" in flat_artifacts_dict:
        image_size = flat_artifacts_dict["image_size"]
    elif "size" in flat_artifacts_dict:
        image_size = flat_artifacts_dict["size"]
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

    if "input_size" in flat_artifacts_dict:
        input_size = flat_artifacts_dict["input_size"]
        shapes["num_channels"] = input_size[0]
        shapes["height"] = input_size[1]
        shapes["width"] = input_size[2]

    # classification labels
    if "id2label" in flat_artifacts_dict:
        id2label = flat_artifacts_dict["id2label"]
        shapes["num_labels"] = len(id2label)
    elif "num_classes" in flat_artifacts_dict:
        shapes["num_labels"] = flat_artifacts_dict["num_classes"]

    # object detection labels
    if "num_queries" in flat_artifacts_dict:
        shapes["num_queries"] = flat_artifacts_dict["num_queries"]

    # image-text input
    if "patch_size" in flat_artifacts_dict:
        shapes["patch_size"] = flat_artifacts_dict["patch_size"]
    if "in_chans" in flat_artifacts_dict:
        shapes["num_channels"] = flat_artifacts_dict["in_chans"]
    if "image_seq_len" in flat_artifacts_dict:
        shapes["image_seq_len"] = flat_artifacts_dict["image_seq_len"]
    if "image_token_id" in flat_artifacts_dict:
        shapes["image_token_id"] = flat_artifacts_dict["image_token_id"]
    if "spatial_merge_size" in flat_artifacts_dict:
        shapes["spatial_merge_size"] = flat_artifacts_dict["spatial_merge_size"]
    if "do_image_splitting" in flat_artifacts_dict:
        shapes["do_image_splitting"] = flat_artifacts_dict["do_image_splitting"]

    if "temporal_patch_size" in flat_artifacts_dict:
        shapes["temporal_patch_size"] = flat_artifacts_dict["temporal_patch_size"]

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


def fast_random_tensor(tensor: "Tensor", *args: Any, **kwargs: Any) -> "Tensor":
    return torch.nn.init.uniform_(tensor)


@contextmanager
def fast_weights_init():
    # Replace the initialization functions
    for name, init_func in TORCH_INIT_FUNCTIONS.items():
        if name != "uniform_":  # avoid recursion
            setattr(torch.nn.init, name, fast_random_tensor)
    try:
        yield
    finally:
        # Restore the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            if name != "uniform_":  # avoid recursion
                setattr(torch.nn.init, name, init_func)
