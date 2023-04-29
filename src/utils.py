from transformers import AutoModelForSequenceClassification, \
    AutoModelForAudioClassification

TASK_TO_AUTOMODEL = {
    "sequence-classification": AutoModelForSequenceClassification,
    "audio-classification": AutoModelForAudioClassification
}
