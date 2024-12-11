TASKS_TO_OVMODELS = {
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
TASKS_TO_OVPIPELINES = {
    "inpainting": "optimum.intel.openvino.OVPipelineForInpainting",
    "text-to-image": "optimum.intel.openvino.OVPipelineForText2Image",
    "image-to-image": "optimum.intel.openvino.OVPipelineForImage2Image",
}
