TASKS_TO_IPEXMODELS = {
    "fill-mask": "optimum.intel.IPEXModelForMaskedLM",
    "text-generation": "optimum.intel.IPEXModelForCausalLM",
    "feature-extraction": "optimum.intel.IPEXModel",
    "text-classification": "optimum.intel.IPEXModelForSequenceClassification",
    "token-classification": "optimum.intel.IPEXModelForTokenClassification",
    "question-answering": "optimum.intel.IPEXModelForQuestionAnswering",
    "image-classification": "optimum.intel.IPEXModelForImageClassification",
    "audio-classification": "optimum.intel.IPEXModelForAudioClassification",
}
