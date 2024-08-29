TASKS_TO_IPEXMODEL = {
    "fill-mask": "optimum.intel.IPEXModelForMaskedLM",
    "text-generation": "optimum.intel.IPEXModelForCausalLM",
    "text2text-generation": "optimum.intel.IPEXModelForSeq2SeqLM",
    "feature-extraction": "optimum.intel.IPEXModelForFeatureExtraction",
    "text-classification": "optimum.intel.IPEXModelForSequenceClassification",
    "token-classification": "optimum.intel.IPEXModelForTokenClassification",
    "question-answering": "optimum.intel.IPEXModelForQuestionAnswering",
    "image-classification": "optimum.intel.IPEXModelForImageClassification",
    "audio-classification": "optimum.intel.IPEXModelForAudioClassification",
    "pix2struct": "optimum.intel.IPEXModelForPix2Struct",
}
TASKS_TO_MODEL_TYPES_TO_IPEXPIPELINE = {
    "text-to-image": {
        "lcm": "optimum.intel.IPEXLatentConsistencyModelPipeline",
        "stable-diffusion": "optimum.intel.IPEXStableDiffusionPipeline",
        "stable-diffusion-xl": "optimum.intel.IPEXStableDiffusionXLPipeline",
    },
}
