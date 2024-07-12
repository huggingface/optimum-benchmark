import importlib
import json
import os
from typing import Optional

import huggingface_hub

_TRANSFORMERS_TASKS_TO_MODEL_LOADERS = {
    # text processing
    "feature-extraction": "AutoModel",
    "fill-mask": "AutoModelForMaskedLM",
    "multiple-choice": "AutoModelForMultipleChoice",
    "question-answering": "AutoModelForQuestionAnswering",
    "token-classification": "AutoModelForTokenClassification",
    "text-classification": "AutoModelForSequenceClassification",
    # audio processing
    "audio-xvector": "AutoModelForAudioXVector",
    "text-to-audio": "AutoModelForTextToSpectrogram",
    "audio-classification": "AutoModelForAudioClassification",
    "audio-frame-classification": "AutoModelForAudioFrameClassification",
    # image processing
    "mask-generation": "AutoModel",
    "image-to-image": "AutoModelForImageToImage",
    "masked-im": "AutoModelForMaskedImageModeling",
    "object-detection": "AutoModelForObjectDetection",
    "depth-estimation": "AutoModelForDepthEstimation",
    "image-segmentation": "AutoModelForImageSegmentation",
    "image-classification": "AutoModelForImageClassification",
    "semantic-segmentation": "AutoModelForSemanticSegmentation",
    "zero-shot-object-detection": "AutoModelForZeroShotObjectDetection",
    "zero-shot-image-classification": "AutoModelForZeroShotImageClassification",
    # text generation
    "image-to-text": "AutoModelForVision2Seq",
    "text-generation": "AutoModelForCausalLM",
    "text2text-generation": "AutoModelForSeq2SeqLM",
    "visual-question-answering": "AutoModelForVisualQuestionAnswering",
    "automatic-speech-recognition": ("AutoModelForSpeechSeq2Seq", "AutoModelForCTC"),
}

_DIFFUSERS_TASKS_TO_MODEL_LOADERS = {
    "inpainting": "AutoPipelineForInpainting",
    "text-to-image": "AutoPipelineForText2Image",
    "image-to-image": "AutoPipelineForImage2Image",
}
_TIMM_TASKS_TO_MODEL_LOADERS = {
    "image-classification": "create_model",
}

_LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP = {
    "timm": _TIMM_TASKS_TO_MODEL_LOADERS,
    "diffusers": _DIFFUSERS_TASKS_TO_MODEL_LOADERS,
    "transformers": _TRANSFORMERS_TASKS_TO_MODEL_LOADERS,
}

_SYNONYM_TASK_MAP = {
    "masked-lm": "fill-mask",
    "causal-lm": "text-generation",
    "default": "feature-extraction",
    "vision2seq-lm": "image-to-text",
    "text-to-speech": "text-to-audio",
    "seq2seq-lm": "text2text-generation",
    "translation": "text2text-generation",
    "summarization": "text2text-generation",
    "mask-generation": "feature-extraction",
    "audio-ctc": "automatic-speech-recognition",
    "sentence-similarity": "feature-extraction",
    "speech2seq-lm": "automatic-speech-recognition",
    "sequence-classification": "text-classification",
    "zero-shot-classification": "text-classification",
}

IMAGE_DIFFUSION_TASKS = [
    "inpainting",
    "text-to-image",
    "image-to-image",
    "stable-diffusion",
    "stable-diffusion-xl",
    "latent-consistency",
]

TEXT_GENERATION_TASKS = [
    "image-to-text",
    "conversational",
    "text-generation",
    "text2text-generation",
    "automatic-speech-recognition",
]

TEXT_EMBEDDING_TASKS = [
    "feature-extraction",
]


def map_from_synonym(task: str) -> str:
    if task in _SYNONYM_TASK_MAP:
        task = _SYNONYM_TASK_MAP[task]
    return task


def infer_library_from_model_name_or_path(model_name_or_path: str, revision: Optional[str] = None) -> str:
    inferred_library_name = None

    is_local = os.path.isdir(model_name_or_path)
    if is_local:
        local_files = os.listdir(model_name_or_path)

        if "model_index.json" in local_files:
            inferred_library_name = "diffusers"
        elif (
            any(file_path.startswith("sentence_") for file_path in local_files)
            or "config_sentence_transformers.json" in local_files
        ):
            inferred_library_name = "sentence_transformers"
        elif "config.json" in local_files:
            config_dict = json.load(open(os.path.join(model_name_or_path, "config.json"), "r"))

            if "pretrained_cfg" in config_dict or "architecture" in config_dict:
                inferred_library_name = "timm"
            elif "_diffusers_version" in config_dict:
                inferred_library_name = "diffusers"
            else:
                inferred_library_name = "transformers"
    else:
        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision)
        inferred_library_name = getattr(model_info, "library_name", None)

    if inferred_library_name is None:
        raise KeyError(f"Could not find the proper library name for {model_name_or_path}.")

    if inferred_library_name == "sentence-transformers":
        # we still don't support sentence-transformers
        inferred_library_name = "transformers"

    return inferred_library_name


def infer_task_from_model_name_or_path(model_name_or_path: str, revision: Optional[str] = None) -> str:
    library_name = infer_library_from_model_name_or_path(model_name_or_path, revision=revision)

    inferred_task_name = None

    if library_name == "timm":
        inferred_task_name = "image-classification"
    elif library_name == "sentence-transformers":
        inferred_task_name = "feature-extraction"
    elif os.path.isdir(model_name_or_path):
        if library_name == "diffusers":
            from diffusers import DiffusionPipeline
            from diffusers.pipelines.auto_pipeline import (
                AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
                AUTO_INPAINT_PIPELINES_MAPPING,
                AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
            )

            config, _ = DiffusionPipeline.load_config(model_name_or_path)
            model_class_name = config["_class_name"]

            for task_name, model_mapping in (
                ("image-to-image", AUTO_IMAGE2IMAGE_PIPELINES_MAPPING),
                ("inpainting", AUTO_INPAINT_PIPELINES_MAPPING),
                ("text-to-image", AUTO_TEXT2IMAGE_PIPELINES_MAPPING),
            ):
                for model_type, model_class in model_mapping.items():
                    if model_class_name == model_class.__name__:
                        inferred_task_name = task_name
                        break
                if inferred_task_name is not None:
                    break
        elif library_name == "transformers":
            from transformers import AutoConfig

            auto_modeling_module = importlib.import_module("transformers.models.auto.modeling_auto")
            config = AutoConfig.from_pretrained(model_name_or_path)
            model_type = config.model_type

            for task_name, model_loaders in _TRANSFORMERS_TASKS_TO_MODEL_LOADERS.items():
                if isinstance(model_loaders, str):
                    model_loaders = (model_loaders,)
                for model_loader in model_loaders:
                    model_loader_class = getattr(auto_modeling_module, model_loader, None)
                    if model_loader_class is not None:
                        model_mapping = model_loader_class._model_mapping._model_mapping
                        if model_type in model_mapping:
                            inferred_task_name = task_name
                            break
    else:
        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision)

        if library_name == "diffusers":
            if model_info.pipeline_tag is not None:
                inferred_task_name = map_from_synonym(model_info.pipeline_tag)
        elif library_name == "transformers":
            if model_info.pipeline_tag is not None:
                inferred_task_name = map_from_synonym(model_info.pipeline_tag)
            else:
                if model_info.transformers_info is not None and model_info.transformersInfo.pipeline_tag is not None:
                    inferred_task_name = map_from_synonym(model_info.transformersInfo.pipeline_tag)
                else:
                    auto_model_class_name = model_info.transformers_info["auto_model"]
                    tasks_to_automodels = _TRANSFORMERS_TASKS_TO_MODEL_LOADERS[model_info.library_name]
                    for task_name, class_name_for_task in tasks_to_automodels.items():
                        if class_name_for_task == auto_model_class_name:
                            inferred_task_name = task_name
                            break
                        inferred_task_name = None

    if inferred_task_name is None:
        raise KeyError(f"Could not find the proper task name for {auto_model_class_name}.")

    return inferred_task_name


def infer_model_type_from_model_name_or_path(model_name_or_path: str, revision: Optional[str] = None) -> str:
    library_name = infer_library_from_model_name_or_path(model_name_or_path, revision=revision)

    inferred_model_type = None

    if library_name == "timm":
        from timm.models import get_pretrained_cfg, load_model_config_from_hf, parse_model_name

        model_source, model_name = parse_model_name(model_name_or_path)
        if model_source == "hf-hub":
            _, model_name = load_model_config_from_hf(model_name)
        config = get_pretrained_cfg(model_name)
        inferred_model_type = config.architecture

    elif library_name == "diffusers":
        from diffusers import DiffusionPipeline
        from diffusers.pipelines.auto_pipeline import (
            AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
            AUTO_INPAINT_PIPELINES_MAPPING,
            AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
        )

        config, _ = DiffusionPipeline.load_config(model_name_or_path)
        model_class_name = config["_class_name"]

        for task_name, model_mapping in (
            ("image-to-image", AUTO_IMAGE2IMAGE_PIPELINES_MAPPING),
            ("inpainting", AUTO_INPAINT_PIPELINES_MAPPING),
            ("text-to-image", AUTO_TEXT2IMAGE_PIPELINES_MAPPING),
        ):
            for model_type, model_class in model_mapping.items():
                if model_class_name == model_class.__name__:
                    inferred_model_type = model_type
                    break
            if inferred_model_type is not None:
                break
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name_or_path)
        inferred_model_type = config.model_type

    if inferred_model_type is None:
        raise KeyError(f"Could not find the proper model type for {model_name_or_path}.")

    return inferred_model_type
