import json
import os
from typing import Optional

from packaging import version

from .hub_utils import HF_API
from .import_utils import diffusers_version, is_diffusers_available, is_torch_available, is_transformers_available

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

if is_transformers_available() and is_torch_available():
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

if is_diffusers_available() and version.parse(diffusers_version()) >= version.parse("0.20.0"):
    try:
        import diffusers.pipelines.auto_pipeline  # type: ignore # noqa: F401
    except Exception as e:
        if "GlmModel" in str(e):
            transformers.GlmModel = None
        else:
            raise e

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
            # diffusers does not have a mappings with just class names
            TASKS_TO_PIPELINE_TYPES_TO_PIPELINE_CLASS_NAMES[task_name][pipeline_type] = pipeline_class.__name__


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
    "sentence-similarity",
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

SYNONYM_LIBRARIES = {
    "sentence-transformers": "transformers",
}


def map_from_synonym_task(task: str) -> str:
    if task in SYNONYM_TASKS:
        task = SYNONYM_TASKS[task]

    return task


def map_from_synonym_library(library: str) -> str:
    if library in SYNONYM_LIBRARIES:
        library = SYNONYM_LIBRARIES[library]

    return library


def is_hf_hub_repo(model_name_or_path: str, token: Optional[str] = None) -> bool:
    try:
        return HF_API.repo_exists(model_name_or_path, token=token)
    except Exception:
        return False


def is_local_dir_repo(model_name_or_path: str) -> bool:
    return os.path.isdir(model_name_or_path)


def get_repo_config(
    model_name_or_path: str,
    config_name: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    if is_hf_hub_repo(model_name_or_path, token=token):
        config = json.load(
            open(
                HF_API.hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=config_name,
                    cache_dir=cache_dir,
                    revision=revision,
                    token=token,
                ),
                mode="r",
            )
        )
    elif is_local_dir_repo(model_name_or_path):
        config = json.load(
            open(
                os.path.join(model_name_or_path, config_name),
                mode="r",
            )
        )
    else:
        raise KeyError(f"`{model_name_or_path}` is neither an hf hub repo nor a local directory.")

    return config


def get_repo_files(model_name_or_path: str, token: Optional[str] = None, revision: Optional[str] = None):
    if is_hf_hub_repo(model_name_or_path, token=token):
        repo_files = HF_API.list_repo_files(model_name_or_path, revision=revision, token=token)
    elif is_local_dir_repo(model_name_or_path):
        repo_files = os.listdir(model_name_or_path)
    else:
        raise KeyError(f"`{model_name_or_path}` is neither an hf hub repo nor a local directory.")

    return repo_files


def infer_library_from_model_name_or_path(
    model_name_or_path: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    inferred_library_name = None

    repo_files = get_repo_files(model_name_or_path, token=token, revision=revision)

    if "model_index.json" in repo_files:
        inferred_library_name = "diffusers"

    elif "config_sentence_transformers.json" in repo_files:
        inferred_library_name = "sentence-transformers"

    elif "config.json" in repo_files:
        config_dict = get_repo_config(
            model_name_or_path, "config.json", token=token, revision=revision, cache_dir=cache_dir
        )

        if "pretrained_cfg" in config_dict:
            inferred_library_name = "timm"
        else:
            inferred_library_name = "transformers"

    elif any(file.endswith(".gguf") or file.endswith(".GGUF") for file in repo_files):
        inferred_library_name = "llama_cpp"

    if inferred_library_name is None:
        raise KeyError(f"Could not find the proper library name for directory {model_name_or_path}.")

    return map_from_synonym_library(inferred_library_name)


def infer_task_from_model_name_or_path(
    model_name_or_path: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    library_name: Optional[str] = None,
) -> str:
    inferred_task_name = None

    if library_name is None:
        library_name = infer_library_from_model_name_or_path(
            model_name_or_path, revision=revision, token=token, cache_dir=cache_dir
        )

    if library_name == "llama_cpp":
        inferred_task_name = "text-generation"

    elif library_name == "timm":
        inferred_task_name = "image-classification"

    elif library_name == "transformers":
        transformers_config = get_repo_config(
            model_name_or_path, "config.json", token=token, revision=revision, cache_dir=cache_dir
        )
        target_class_name = transformers_config["architectures"][0]

        for task_name, model_mapping in TASKS_TO_MODEL_TYPES_TO_MODEL_CLASS_NAMES.items():
            for _, model_class_name in model_mapping.items():
                if target_class_name == model_class_name:
                    inferred_task_name = task_name
                    break
            if inferred_task_name is not None:
                break

        if inferred_task_name is None:
            raise KeyError(f"Could not find the proper task name for target class name {target_class_name}.")

    elif library_name == "diffusers":
        diffusers_config = get_repo_config(
            model_name_or_path, "model_index.json", token=token, revision=revision, cache_dir=cache_dir
        )
        target_class_name = diffusers_config["_class_name"]

        for task_name, pipeline_mapping in TASKS_TO_PIPELINE_TYPES_TO_PIPELINE_CLASS_NAMES.items():
            for _, pipeline_class_name in pipeline_mapping.items():
                if target_class_name == pipeline_class_name or (pipeline_class_name in target_class_name):
                    inferred_task_name = task_name
                    break
            if inferred_task_name is not None:
                break

        if inferred_task_name is None:
            raise KeyError(f"Could not find the proper task name for target class name {target_class_name}.")

    return map_from_synonym_task(inferred_task_name)


def infer_model_type_from_model_name_or_path(
    model_name_or_path: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    library_name: Optional[str] = None,
) -> str:
    inferred_model_type = None

    if library_name is None:
        library_name = infer_library_from_model_name_or_path(
            model_name_or_path, revision=revision, token=token, cache_dir=cache_dir
        )

    if library_name == "llama_cpp":
        inferred_model_type = "llama_cpp"

    elif library_name == "timm":
        timm_config = get_repo_config(
            model_name_or_path, "config.json", token=token, revision=revision, cache_dir=cache_dir
        )
        inferred_model_type = timm_config["architecture"]

    elif library_name == "transformers":
        transformers_config = get_repo_config(
            model_name_or_path, "config.json", token=token, revision=revision, cache_dir=cache_dir
        )
        inferred_model_type = transformers_config["model_type"]

    elif library_name == "diffusers":
        diffusers_config = get_repo_config(
            model_name_or_path, "model_index.json", token=token, revision=revision, cache_dir=cache_dir
        )
        target_class_name = diffusers_config["_class_name"]

        for _, pipeline_mapping in TASKS_TO_PIPELINE_TYPES_TO_PIPELINE_CLASS_NAMES.items():
            for pipeline_type, pipeline_class_name in pipeline_mapping.items():
                if target_class_name == pipeline_class_name or (pipeline_class_name in target_class_name):
                    inferred_model_type = pipeline_type
                    break
            if inferred_model_type is not None:
                break

        if inferred_model_type is None:
            # we use the class name in this case
            inferred_model_type = target_class_name.replace("DiffusionPipeline", "").replace("Pipeline", "")

    return inferred_model_type
