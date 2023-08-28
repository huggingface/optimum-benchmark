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

# let's leave this here for now, it's a good list of tasks supported by transformers
ALL_TASKS = [
    "conversational",
    "feature-extraction",
    "fill-mask",
    "text-generation",
    "text2text-generation",
    "text-classification",
    "token-classification",
    "multiple-choice",
    "object-detection",
    "question-answering",
    "image-classification",
    "image-segmentation",
    "mask-generation",
    "masked-im",
    "semantic-segmentation",
    "automatic-speech-recognition",
    "audio-classification",
    "audio-frame-classification",
    "audio-xvector",
    "image-to-text",
    "stable-diffusion",
    "stable-diffusion-xl",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
]


def infer_task(model: str, revision: str) -> str:
    from optimum.exporters import TasksManager

    return TasksManager.infer_task_from_model(
        model=model,
        revision=revision,
    )
