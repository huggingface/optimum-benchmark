import importlib
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
    "conversational": ("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"),
    # image processing
    "mask-generation": "AutoModel",
    "image-to-image": "AutoModelForImageToImage",
    "masked-im": "AutoModelForMaskedImageModeling",
    "object-detection": "AutoModelForObjectDetection",
    "depth-estimation": "AutoModelForDepthEstimation",
    "image-classification": "AutoModelForImageClassification",
    "semantic-segmentation": "AutoModelForSemanticSegmentation",
    "zero-shot-object-detection": "AutoModelForZeroShotObjectDetection",
    "zero-shot-image-classification": "AutoModelForZeroShotImageClassification",
    "image-segmentation": ("AutoModelForImageSegmentation", "AutoModelForSemanticSegmentation"),
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
    "stable-diffusion": "StableDiffusionPipeline",  # should be deprecated
    "stable-diffusion-xl": "StableDiffusionXLImg2ImgPipeline",  # should be deprecated
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
    "causal-lm-with-past": "text-generation-with-past",
    "default-with-past": "feature-extraction-with-past",
    "seq2seq-lm-with-past": "text2text-generation-with-past",
    "speech2seq-lm-with-past": "automatic-speech-recognition-with-past",
}

IMAGE_DIFFUSION_TASKS = [
    "inpainting",
    "text-to-image",
    "image-to-image",
    "stable-diffusion",
    "stable-diffusion-xl",
]

TEXT_GENERATION_TASKS = [
    "image-to-text",
    "conversational",
    "text-generation",
    "text2text-generation",
    "automatic-speech-recognition",
]

TEXT_EMBEDDING_TASKS = [
    "fill-mask",
    "feature-extraction",
]


def map_from_synonym(task: str) -> str:
    if task in _SYNONYM_TASK_MAP:
        task = _SYNONYM_TASK_MAP[task]
    return task


def infer_library_from_model_name_or_path(model_name_or_path: str, revision: Optional[str] = None) -> str:
    is_local = os.path.isdir(model_name_or_path)

    if is_local:
        raise RuntimeError("Cannot infer the library from a local directory yet, please specify the library manually.")

    model_info = huggingface_hub.model_info(model_name_or_path, revision=revision)

    inferred_library_name = getattr(model_info, "library_name", None)

    if inferred_library_name is None:
        raise KeyError(f"Could not find the proper library name for {model_name_or_path}.")

    if inferred_library_name == "sentence-transformers":
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
        if "text-to-image" in model_info.tags:
            inferred_task_name = "text-to-image"
        elif "image-to-image" in model_info.tags:
            inferred_task_name = "image-to-image"
        elif "inpainting" in model_info.tags:
            inferred_task_name = "inpainting"
        else:
            class_name = model_info.config["diffusers"]["class_name"]
            inferred_task_name = "stable-diffusion-xl" if "XL" in class_name else "stable-diffusion"

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


# adapted from https://github.com/huggingface/optimum/blob/main/optimum/exporters/tasks.py without torch dependency
def get_automodel_class_for_task(
    task: str,
    auto_model_class_name: Optional[str] = None,
    model_type: Optional[str] = None,
    library: str = "transformers",
    framework: str = "pt",
):
    task = map_from_synonym(task)

    if framework == "pt":
        tasks_to_model_loader = _LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP[library]
    elif framework == "jax":
        raise NotImplementedError("JAX is not supported yet.")
    elif framework == "tf":
        raise NotImplementedError("TensorFlow is not supported yet.")
    else:
        raise NotImplementedError("Only PyTorch is supported for now.")

    loaded_library = importlib.import_module(library)

    if auto_model_class_name is None:
        if task not in tasks_to_model_loader:
            raise KeyError(
                f"Unknown task: {task}. Possible values are: "
                + ", ".join([f"`{key}` for {tasks_to_model_loader[key]}" for key in tasks_to_model_loader])
            )

        if isinstance(tasks_to_model_loader[task], str):
            inferred_auto_model_class_name = tasks_to_model_loader[task]
        elif isinstance(tasks_to_model_loader[task], tuple):
            if model_type is None:
                inferred_auto_model_class_name = tasks_to_model_loader[task][0]
            else:
                for auto_class_name in tasks_to_model_loader[task]:
                    model_mapping = getattr(loaded_library, auto_class_name)._model_mapping._model_mapping

                    if model_type in model_mapping or model_type.replace("-", "_") in model_mapping:
                        inferred_auto_model_class_name = auto_class_name
                        break

                    inferred_auto_model_class_name = None

    if inferred_auto_model_class_name is None:
        raise ValueError(f"Could not find the model class name for task {task}.")

    inferred_model_class = getattr(loaded_library, inferred_auto_model_class_name)

    return inferred_model_class
