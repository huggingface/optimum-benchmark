import importlib
import json
import os
from typing import Optional

import huggingface_hub

from .backends.diffusers_utils import get_diffusers_pretrained_config
from .backends.timm_utils import get_timm_pretrained_config
from .backends.transformers_utils import get_transformers_pretrained_config
from .import_utils import is_diffusers_available, is_torch_available

TASKS_TO_AUTO_MODEL_CLASS_NAMES = {
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
    "image-text-to-text": "AutoModelForImageTextToText",
    "visual-question-answering": "AutoModelForVisualQuestionAnswering",
    "automatic-speech-recognition": ("AutoModelForSpeechSeq2Seq", "AutoModelForCTC"),
}

TASKS_TO_AUTO_PIPELINE_CLASS_NAMES = {
    "inpainting": "AutoPipelineForInpainting",
    "text-to-image": "AutoPipelineForText2Image",
    "image-to-image": "AutoPipelineForImage2Image",
}

TASKS_TO_MODEL_TYPES_TO_MODEL_CLASS_NAMES = {}

if is_torch_available():
    import transformers

    for task_name, auto_model_class_names in TASKS_TO_AUTO_MODEL_CLASS_NAMES.items():
        TASKS_TO_MODEL_TYPES_TO_MODEL_CLASS_NAMES[task_name] = {}

        if isinstance(auto_model_class_names, str):
            auto_model_class_names = (auto_model_class_names,)

        for auto_model_class_name in auto_model_class_names:
            auto_model_class = getattr(transformers, auto_model_class_name, None)
            if auto_model_class is not None:
                TASKS_TO_MODEL_TYPES_TO_MODEL_CLASS_NAMES[task_name].update(
                    auto_model_class._model_mapping._model_mapping
                )


TASKS_TO_PIPELINE_TYPES_TO_PIPELINE_CLASS_NAMES = {}

if is_diffusers_available():
    import diffusers

    if hasattr(diffusers, "pipelines") and hasattr(diffusers.pipelines, "auto_pipeline"):
        from diffusers.pipelines.auto_pipeline import (
            AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
            AUTO_INPAINT_PIPELINES_MAPPING,
            AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
        )

        TASKS_TO_PIPELINE_TYPES_TO_PIPELINE_CLASS_NAMES = {
            "inpainting": AUTO_INPAINT_PIPELINES_MAPPING.copy(),
            "text-to-image": AUTO_TEXT2IMAGE_PIPELINES_MAPPING.copy(),
            "image-to-image": AUTO_IMAGE2IMAGE_PIPELINES_MAPPING.copy(),
        }

        for task_name, pipeline_mapping in TASKS_TO_PIPELINE_TYPES_TO_PIPELINE_CLASS_NAMES.items():
            for pipeline_type, pipeline_class in pipeline_mapping.items():
                TASKS_TO_PIPELINE_TYPES_TO_PIPELINE_CLASS_NAMES[task_name][pipeline_type] = pipeline_class.__name__
    else:
        TASKS_TO_PIPELINE_TYPES_TO_PIPELINE_CLASS_NAMES = {}


IMAGE_DIFFUSION_TASKS = [
    "inpainting",
    "text-to-image",
    "image-to-image",
]

TEXT_GENERATION_TASKS = [
    "image-to-text",
    "conversational",
    "text-generation",
    "image-text-to-text",
    "text2text-generation",
    "automatic-speech-recognition",
]

TEXT_EMBEDDING_TASKS = [
    "feature-extraction",
]

SYNONYM_TASKS = {
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


def map_from_synonym(task: str) -> str:
    if task in SYNONYM_TASKS:
        task = SYNONYM_TASKS[task]
    return task


def infer_library_from_model_name_or_path(
    model_name_or_path: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    inferred_library_name = None

    # if model_name_or_path is a repo
    if huggingface_hub.repo_exists(model_name_or_path, token=token):
        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision, token=token)
        inferred_library_name = getattr(model_info, "library_name", None)

        if inferred_library_name is None:
            repo_files = huggingface_hub.list_repo_files(model_name_or_path, revision=revision, token=token)
            if "model_index.json" in repo_files:
                inferred_library_name = "diffusers"
            elif "config.json" in repo_files:
                config_dict = json.loads(
                    huggingface_hub.hf_hub_download(
                        repo_id=model_name_or_path, filename="config.json", revision=revision, token=token
                    )
                )
                if "pretrained_cfg" in config_dict or "architecture" in config_dict:
                    inferred_library_name = "timm"
                elif "_diffusers_version" in config_dict:
                    inferred_library_name = "diffusers"
                else:
                    inferred_library_name = "transformers"

        if inferred_library_name is None:
            raise RuntimeError(f"Could not infer library name from repo {model_name_or_path}.")

    # if model_name_or_path is a directory
    elif os.path.isdir(model_name_or_path):
        local_files = os.listdir(model_name_or_path)

        if "model_index.json" in local_files:
            inferred_library_name = "diffusers"
        elif "config.json" in local_files:
            config_dict = json.load(open(os.path.join(model_name_or_path, "config.json"), "r"))

            if "pretrained_cfg" in config_dict or "architecture" in config_dict:
                inferred_library_name = "timm"
            elif "_diffusers_version" in config_dict:
                inferred_library_name = "diffusers"
            else:
                inferred_library_name = "transformers"

        if inferred_library_name is None:
            raise KeyError(f"Could not find the proper library name for directory {model_name_or_path}.")

    else:
        raise KeyError(
            f"Could not find the proper library name for {model_name_or_path}"
            " because it's neither a repo nor a directory."
        )

    # for now, we still use transformers for sentence-transformers
    if inferred_library_name == "sentence-transformers":
        inferred_library_name = "transformers"

    return inferred_library_name


def infer_task_from_model_name_or_path(
    model_name_or_path: str,
    library_name: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    if library_name is None:
        library_name = infer_library_from_model_name_or_path(model_name_or_path, revision=revision, token=token)

    inferred_task_name = None

    if library_name == "timm":
        inferred_task_name = "image-classification"

    elif library_name == "sentence-transformers":
        inferred_task_name = "feature-extraction"

    elif huggingface_hub.repo_exists(model_name_or_path, token=token):
        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision, token=token)

        if model_info.pipeline_tag is not None:
            inferred_task_name = map_from_synonym(model_info.pipeline_tag)

        elif inferred_task_name is None:
            if model_info.transformers_info is not None and model_info.transformersInfo.pipeline_tag is not None:
                inferred_task_name = map_from_synonym(model_info.transformersInfo.pipeline_tag)
            else:
                target_auto_model = model_info.transformers_info["auto_model"]
                for task_name, auto_model_class_names in TASKS_TO_AUTO_MODEL_CLASS_NAMES.items():
                    if isinstance(auto_model_class_names, str):
                        auto_model_class_names = (auto_model_class_names,)

                    for auto_model_class_name in auto_model_class_names:
                        if target_auto_model == auto_model_class_name:
                            inferred_task_name = task_name
                            break
                    if inferred_task_name is not None:
                        break

    elif os.path.isdir(model_name_or_path):
        if library_name == "diffusers":
            diffusers_config = get_diffusers_pretrained_config(model_name_or_path, revision=revision, token=token)
            target_class_name = diffusers_config["_class_name"]

            for task_name, pipeline_mapping in TASKS_TO_PIPELINE_TYPES_TO_PIPELINE_CLASS_NAMES.items():
                for _, pipeline_class_name in pipeline_mapping.items():
                    if target_class_name == pipeline_class_name:
                        inferred_task_name = task_name
                        break
                if inferred_task_name is not None:
                    break

        elif library_name == "transformers":
            transformers_config = get_transformers_pretrained_config(model_name_or_path, revision=revision, token=token)
            auto_modeling_module = importlib.import_module("transformers.models.auto.modeling_auto")
            model_type = transformers_config.model_type

            for task_name, model_loaders in TASKS_TO_MODEL_TYPES_TO_MODEL_CLASS_NAMES.items():
                if isinstance(model_loaders, str):
                    model_loaders = (model_loaders,)
                for model_loader in model_loaders:
                    model_loader_class = getattr(auto_modeling_module, model_loader)
                    model_mapping = model_loader_class._model_mapping._model_mapping
                    if model_type in model_mapping:
                        inferred_task_name = task_name
                        break
                if inferred_task_name is not None:
                    break

    if inferred_task_name is None:
        raise KeyError(f"Could not find the proper task name for {auto_model_class_name}.")

    return inferred_task_name


def infer_model_type_from_model_name_or_path(
    model_name_or_path: str,
    library_name: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> str:
    if library_name is None:
        library_name = infer_library_from_model_name_or_path(model_name_or_path, revision=revision, token=token)

    inferred_model_type = None

    if library_name == "llama_cpp":
        inferred_model_type = "llama_cpp"

    elif library_name == "timm":
        timm_config = get_timm_pretrained_config(model_name_or_path)
        inferred_model_type = timm_config.architecture

    elif library_name == "diffusers":
        config = get_diffusers_pretrained_config(model_name_or_path, revision=revision, token=token)
        target_class_name = config["_class_name"]

        for _, pipeline_mapping in TASKS_TO_PIPELINE_TYPES_TO_PIPELINE_CLASS_NAMES.items():
            for pipeline_type, pipeline_class_name in pipeline_mapping.items():
                if target_class_name == pipeline_class_name:
                    inferred_model_type = pipeline_type
                    break
            if inferred_model_type is not None:
                break

    else:
        transformers_config = get_transformers_pretrained_config(
            model_name_or_path, revision=revision, token=token, trust_remote_code=trust_remote_code
        )
        inferred_model_type = transformers_config.model_type

    if inferred_model_type is None:
        raise KeyError(f"Could not find the proper model type for {model_name_or_path}.")

    return inferred_model_type
