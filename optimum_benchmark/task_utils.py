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


# adapted from https://github.com/huggingface/optimum/blob/main/optimum/exporters/tasks.py without torch dependency
def infer_task_from_model_name_or_path(model_name_or_path: str, revision: Optional[str] = None) -> str:
    is_local = os.path.isdir(model_name_or_path)

    if is_local:
        raise RuntimeError("Cannot infer the task from a local directory yet, please specify the task manually.")

    model_info = huggingface_hub.model_info(model_name_or_path, revision=revision)
    library_name = infer_library_from_model_name_or_path(model_name_or_path, revision=revision)

    if library_name == "timm":
        inferred_task_name = "image-classification"

    elif library_name == "sentence-transformers":
        inferred_task_name = "feature-extraction"

    elif library_name == "diffusers":
        if model_info.pipeline_tag is not None:
            inferred_task_name = map_from_synonym(model_info.pipeline_tag)
        else:
            inferred_task_name = "text-to-image"

    elif library_name == "transformers":
        if model_info.pipeline_tag is not None:
            inferred_task_name = map_from_synonym(model_info.pipeline_tag)
        else:
            pipeline_tag = model_info.transformersInfo.pipeline_tag

            if model_info.transformers_info is not None and pipeline_tag is not None:
                inferred_task_name = map_from_synonym(pipeline_tag)
            else:
                auto_model_class_name = model_info.transformers_info["auto_model"]
                tasks_to_automodels = _LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP[model_info.library_name]
                for task_name, class_name_for_task in tasks_to_automodels.items():
                    if class_name_for_task == auto_model_class_name:
                        inferred_task_name = task_name
                        break
                    inferred_task_name = None

    else:
        raise NotImplementedError(f"Library {library_name} is not supported yet.")

    if inferred_task_name is None:
        raise KeyError(f"Could not find the proper task name for {auto_model_class_name}.")

    return inferred_task_name
