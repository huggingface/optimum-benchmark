import importlib
import os
from typing import Optional

import huggingface_hub

from .import_utils import is_tensorflow_available

_TRANSFORMERS_TASKS_TO_MODEL_LOADERS = {
    "audio-classification": "AutoModelForAudioClassification",
    "audio-frame-classification": "AutoModelForAudioFrameClassification",
    "audio-xvector": "AutoModelForAudioXVector",
    "automatic-speech-recognition": ("AutoModelForSpeechSeq2Seq", "AutoModelForCTC"),
    "conversational": ("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"),
    "depth-estimation": "AutoModelForDepthEstimation",
    "feature-extraction": "AutoModel",
    "fill-mask": "AutoModelForMaskedLM",
    "image-classification": "AutoModelForImageClassification",
    "image-segmentation": ("AutoModelForImageSegmentation", "AutoModelForSemanticSegmentation"),
    "image-to-image": "AutoModelForImageToImage",
    "image-to-text": "AutoModelForVision2Seq",
    "mask-generation": "AutoModel",
    "masked-im": "AutoModelForMaskedImageModeling",
    "multiple-choice": "AutoModelForMultipleChoice",
    "object-detection": "AutoModelForObjectDetection",
    "question-answering": "AutoModelForQuestionAnswering",
    "semantic-segmentation": "AutoModelForSemanticSegmentation",
    "text-to-audio": "AutoModelForTextToSpectrogram",
    "text-generation": "AutoModelForCausalLM",
    "text2text-generation": "AutoModelForSeq2SeqLM",
    "text-classification": "AutoModelForSequenceClassification",
    "token-classification": "AutoModelForTokenClassification",
    "zero-shot-image-classification": "AutoModelForZeroShotImageClassification",
    "zero-shot-object-detection": "AutoModelForZeroShotObjectDetection",
}
_TRANSFORMERS_TASKS_TO_TF_MODEL_LOADERS = {
    "conversational": ("TFAutoModelForCausalLM", "TFAutoModelForSeq2SeqLM"),
    "document-question-answering": "TFAutoModelForDocumentQuestionAnswering",
    "feature-extraction": "TFAutoModel",
    "fill-mask": "TFAutoModelForMaskedLM",
    "text-generation": "TFAutoModelForCausalLM",
    "image-classification": "TFAutoModelForImageClassification",
    "text2text-generation": "TFAutoModelForSeq2SeqLM",
    "text-classification": "TFAutoModelForSequenceClassification",
    "token-classification": "TFAutoModelForTokenClassification",
    "multiple-choice": "TFAutoModelForMultipleChoice",
    "object-detection": "TFAutoModelForObjectDetection",
    "question-answering": "TFAutoModelForQuestionAnswering",
    "image-segmentation": "TFAutoModelForImageSegmentation",
    "masked-im": "TFAutoModelForMaskedImageModeling",
    "semantic-segmentation": "TFAutoModelForSemanticSegmentation",
    "automatic-speech-recognition": "TFAutoModelForSpeechSeq2Seq",
    "audio-classification": "TFAutoModelForAudioClassification",
    "audio-frame-classification": "TFAutoModelForAudioFrameClassification",
    "audio-xvector": "TFAutoModelForAudioXVector",
    "image-to-text": "TFAutoModelForVision2Seq",
    "zero-shot-image-classification": "TFAutoModelForZeroShotImageClassification",
    "zero-shot-object-detection": "TFAutoModelForZeroShotObjectDetection",
}
_DIFFUSERS_TASKS_TO_MODEL_LOADERS = {
    "stable-diffusion": "AutoPipelineForText2Image",
    "stable-diffusion-xl": "AutoPipelineForText2Image",
}
_TIMM_TASKS_TO_MODEL_LOADERS = {"image-classification": "create_model"}
_LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP = {"transformers": _TRANSFORMERS_TASKS_TO_TF_MODEL_LOADERS}
_LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP = {
    "transformers": _TRANSFORMERS_TASKS_TO_MODEL_LOADERS,
    "diffusers": _DIFFUSERS_TASKS_TO_MODEL_LOADERS,
    "timm": _TIMM_TASKS_TO_MODEL_LOADERS,
}
_SYNONYM_TASK_MAP = {
    "audio-ctc": "automatic-speech-recognition",
    "causal-lm": "text-generation",
    "causal-lm-with-past": "text-generation-with-past",
    "default": "feature-extraction",
    "default-with-past": "feature-extraction-with-past",
    "masked-lm": "fill-mask",
    "mask-generation": "feature-extraction",
    "sentence-similarity": "feature-extraction",
    "seq2seq-lm": "text2text-generation",
    "seq2seq-lm-with-past": "text2text-generation-with-past",
    "sequence-classification": "text-classification",
    "speech2seq-lm": "automatic-speech-recognition",
    "speech2seq-lm-with-past": "automatic-speech-recognition-with-past",
    "summarization": "text2text-generation",
    "text-to-speech": "text-to-audio",
    "translation": "text2text-generation",
    "vision2seq-lm": "image-to-text",
    "zero-shot-classification": "text-classification",
}
_CUSTOM_CLASSES = {
    ("pt", "pix2struct", "image-to-text"): ("transformers", "Pix2StructForConditionalGeneration"),
    ("pt", "pix2struct", "visual-question-answering"): ("transformers", "Pix2StructForConditionalGeneration"),
    ("pt", "visual-bert", "question-answering"): ("transformers", "VisualBertForQuestionAnswering"),
    ("pt", "vision-encoder-decoder", "document-question-answering"): ("transformers", "VisionEncoderDecoderModel"),
}

TEXT_EMBEDDING_TASKS = ["feature-extraction", "fill-mask"]

IMAGE_DIFFUSION_TASKS = ["stable-diffusion", "stable-diffusion-xl"]

TEXT_GENERATION_TASKS = ["image-to-text", "text-generation", "text2text-generation", "automatic-speech-recognition"]

IMAGE_PROCESSING_TASKS = [
    "image-classification",
    "image-segmentation",
    "object-detection",
    "semantic-segmentation",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
]

def map_from_synonym(task: str) -> str:
    if task in _SYNONYM_TASK_MAP:
        task = _SYNONYM_TASK_MAP[task]
    return task


def infer_library_from_model_name_or_path(model_name_or_path: str, revision: Optional[str] = None) -> str:
    inferred_library_name = None
    is_local = os.path.isdir(model_name_or_path)

    if is_local:
        raise RuntimeError("Cannot infer the library from a local directory yet, please specify the library manually.")
    else:
        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision)
        inferred_library_name = model_info.library_name

    if inferred_library_name is None:
        raise KeyError(f"Could not find the proper library name for {model_name_or_path}.")

    return inferred_library_name


# adapted from https://github.com/huggingface/optimum/blob/main/optimum/exporters/tasks.py without torch dependency
def infer_task_from_model_name_or_path(model_name_or_path: str, revision: Optional[str] = None) -> str:
    inferred_task_name = None
    is_local = os.path.isdir(model_name_or_path)

    if is_local:
        raise RuntimeError("Cannot infer the task from a local directory yet, please specify the task manually.")
    else:
        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision)
        if getattr(model_info, "library_name", None) == "diffusers":
            class_name = model_info.config["diffusers"]["class_name"]
            inferred_task_name = "stable-diffusion-xl" if "StableDiffusionXL" in class_name else "stable-diffusion"
        elif getattr(model_info, "library_name", None) == "timm":
            inferred_task_name = "image-classification"
        else:
            pipeline_tag = getattr(model_info, "pipeline_tag", None)
            # conversational is not a supported task per se, just an alias that may map to text-generaton or text2text-generation
            if pipeline_tag is not None and pipeline_tag not in ["conversational", "object-detection"]:
                inferred_task_name = map_from_synonym(model_info.pipeline_tag)
            else:
                transformers_info = model_info.transformersInfo
                if transformers_info is not None and transformers_info.get("pipeline_tag") is not None:
                    inferred_task_name = map_from_synonym(transformers_info["pipeline_tag"])
                else:
                    # transformersInfo does not always have a pipeline_tag attribute
                    class_name_prefix = ""
                    if is_tensorflow_available():
                        tasks_to_automodels = _LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP[model_info.library_name]
                        class_name_prefix = "TF"
                    else:
                        tasks_to_automodels = _LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP[model_info.library_name]

                    auto_model_class_name = transformers_info["auto_model"]
                    if not auto_model_class_name.startswith("TF"):
                        auto_model_class_name = f"{class_name_prefix}{auto_model_class_name}"
                    for task_name, class_name_for_task in tasks_to_automodels.items():
                        if class_name_for_task == auto_model_class_name:
                            inferred_task_name = task_name
                            break

    if inferred_task_name is None:
        raise KeyError(f"Could not find the proper task name for {auto_model_class_name}.")

    return inferred_task_name


# adapted from https://github.com/huggingface/optimum/blob/main/optimum/exporters/tasks.py without torch dependency
def get_automodel_class_for_task(
    task: str,
    model_class_name: Optional[str] = None,
    model_type: Optional[str] = None,
    library: str = "transformers",
    framework: str = "pt",
):
    task = task.replace("-with-past", "")
    task = map_from_synonym(task)

    if (framework, model_type, task) in _CUSTOM_CLASSES:
        library, class_name = _CUSTOM_CLASSES[(framework, model_type, task)]
        loaded_library = importlib.import_module(library)

        return getattr(loaded_library, class_name)
    else:
        if framework == "pt":
            tasks_to_model_loader = _LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP[library]
        else:
            raise NotImplementedError("Only PyTorch is supported for now.")

        loaded_library = importlib.import_module(library)

        if model_class_name is None:
            if task not in tasks_to_model_loader:
                raise KeyError(
                    f"Unknown task: {task}. Possible values are: "
                    + ", ".join([f"`{key}` for {tasks_to_model_loader[key]}" for key in tasks_to_model_loader])
                )

            if isinstance(tasks_to_model_loader[task], str):
                model_class_name = tasks_to_model_loader[task]
            else:
                # automatic-speech-recognition case, which may map to several auto class
                if library == "transformers":
                    if model_type is None:
                        model_class_name = tasks_to_model_loader[task][0]
                    else:
                        for autoclass_name in tasks_to_model_loader[task]:
                            module = getattr(loaded_library, autoclass_name)
                            if (
                                model_type in module._model_mapping._model_mapping
                                or model_type.replace("-", "_") in module._model_mapping._model_mapping
                            ):
                                model_class_name = autoclass_name
                                break

                        if model_class_name is None:
                            raise ValueError(
                                f"Unrecognized configuration classes {tasks_to_model_loader[task]} do not match"
                                f" with the model type {model_type} and task {task}."
                            )
                else:
                    raise NotImplementedError(
                        "For library other than transformers, the _TASKS_TO_MODEL_LOADER mapping should be one to one."
                    )

        return getattr(loaded_library, model_class_name)
