import os
from typing import Optional

import huggingface_hub

# constants from https://github.com/huggingface/optimum/blob/main/optimum/exporters/tasks.py
TASKS_TO_AUTOMODELS = {
    "conversational": ("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"),
    "feature-extraction": "AutoModel",
    "fill-mask": "AutoModelForMaskedLM",
    "text-generation": "AutoModelForCausalLM",
    "text2text-generation": "AutoModelForSeq2SeqLM",
    "text-classification": "AutoModelForSequenceClassification",
    "token-classification": "AutoModelForTokenClassification",
    "multiple-choice": "AutoModelForMultipleChoice",
    "object-detection": "AutoModelForObjectDetection",
    "question-answering": "AutoModelForQuestionAnswering",
    "image-classification": "AutoModelForImageClassification",
    "image-segmentation": ("AutoModelForImageSegmentation", "AutoModelForSemanticSegmentation"),
    "mask-generation": "AutoModel",
    "masked-im": "AutoModelForMaskedImageModeling",
    "semantic-segmentation": "AutoModelForSemanticSegmentation",
    "automatic-speech-recognition": ("AutoModelForSpeechSeq2Seq", "AutoModelForCTC"),
    "audio-classification": "AutoModelForAudioClassification",
    "audio-frame-classification": "AutoModelForAudioFrameClassification",
    "audio-xvector": "AutoModelForAudioXVector",
    "image-to-text": "AutoModelForVision2Seq",
    "stable-diffusion": "StableDiffusionPipeline",
    "stable-diffusion-xl": "StableDiffusionXLImg2ImgPipeline",
    "zero-shot-image-classification": "AutoModelForZeroShotImageClassification",
    "zero-shot-object-detection": "AutoModelForZeroShotObjectDetection",
}
SYNONYM_TASK_MAP = {
    "sequence-classification": "text-classification",
    "causal-lm": "text-generation",
    "causal-lm-with-past": "text-generation-with-past",
    "seq2seq-lm": "text2text-generation",
    "seq2seq-lm-with-past": "text2text-generation-with-past",
    "speech2seq-lm": "automatic-speech-recognition",
    "speech2seq-lm-with-past": "automatic-speech-recognition-with-past",
    "masked-lm": "fill-mask",
    "mask-generation": "feature-extraction",
    "vision2seq-lm": "image-to-text",
    "default": "feature-extraction",
    "default-with-past": "feature-extraction-with-past",
    "audio-ctc": "automatic-speech-recognition",
    "translation": "text2text-generation",
    "sentence-similarity": "feature-extraction",
    "summarization": "text2text-generation",
    "zero-shot-classification": "text-classification",
}
DIFFUSION_TASKS = [
    "stable-diffusion",
    "stable-diffusion-xl",
]

TEXT_GENERATION_TASKS = [
    "image-to-text",
    "text-generation",
    "text2text-generation",
    "automatic-speech-recognition",
]


def map_from_synonym(task: str) -> str:
    if task in SYNONYM_TASK_MAP:
        task = SYNONYM_TASK_MAP[task]
    return task


# adapted from https://github.com/huggingface/optimum/blob/main/optimum/exporters/tasks.py without torch dependency
def infer_task_from_model_name_or_path(
    model_name_or_path: str, subfolder: str = "", revision: Optional[str] = None
) -> str:
    inferred_task_name = None
    is_local = os.path.isdir(os.path.join(model_name_or_path, subfolder))

    if is_local:
        # TODO: maybe implement that.
        raise RuntimeError("Cannot infer the task from a local directory yet, please specify the task manually.")
    else:
        if subfolder != "":
            raise RuntimeError(
                "Cannot infer the task from a model repo with a subfolder yet, please specify the task manually."
            )

        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision)
        if model_info.library_name == "diffusers":
            # TODO : getattr(model_info, "model_index") defining auto_model_class_name currently set to None
            for task in ("stable-diffusion-xl", "stable-diffusion"):
                if task in model_info.tags:
                    inferred_task_name = task
                    break
        else:
            pipeline_tag = getattr(model_info, "pipeline_tag", None)
            # conversational is not a supported task per se, just an alias that may map to
            # text-generaton or text2text-generation
            if pipeline_tag is not None and pipeline_tag != "conversational":
                inferred_task_name = map_from_synonym(model_info.pipeline_tag)
            else:
                transformers_info = model_info.transformersInfo
                if transformers_info is not None and transformers_info.get("pipeline_tag") is not None:
                    inferred_task_name = map_from_synonym(transformers_info["pipeline_tag"])
                else:
                    # transformersInfo does not always have a pipeline_tag attribute
                    auto_model_class_name = transformers_info["auto_model"]
                    for task_name, class_name_for_task in TASKS_TO_AUTOMODELS.items():
                        if class_name_for_task == auto_model_class_name:
                            inferred_task_name = task_name
                            break

    if inferred_task_name is None:
        raise KeyError(f"Could not find the proper task name for {auto_model_class_name}.")

    return inferred_task_name
