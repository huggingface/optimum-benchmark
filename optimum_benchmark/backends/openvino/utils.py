TASKS_TO_OVMODEL = {
    "fill-mask": "optimum.intel.openvino.OVModelForMaskedLM",
    "text-generation": "optimum.intel.openvino.OVModelForCausalLM",
    "text2text-generation": "optimum.intel.openvino.OVModelForSeq2SeqLM",
    "feature-extraction": "optimum.intel.openvino.OVModelForFeatureExtraction",
    "text-classification": "optimum.intel.openvino.OVModelForSequenceClassification",
    "token-classification": "optimum.intel.openvino.OVModelForTokenClassification",
    "question-answering": "optimum.intel.openvino.OVModelForQuestionAnswering",
    "image-classification": "optimum.intel.openvino.OVModelForImageClassification",
    "audio-classification": "optimum.intel.openvino.OVModelForAudioClassification",
    "pix2struct": "optimum.intel.openvino.OVModelForPix2Struct",
}
TASKS_TO_MODEL_TYPES_TO_OVPIPELINE = {
    "text-to-image": {
        "lcm": "optimum.intel.openvino.OVLatentConsistencyModelPipeline",
        "stable-diffusion": "optimum.intel.openvino.OVStableDiffusionPipeline",
        "stable-diffusion-xl": "optimum.intel.openvino.OVStableDiffusionXLPipeline",
    },
}
