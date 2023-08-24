DEFAULT_QUANTIZATION_CONFIG = {
    "compression": None,
    "input_info": None,
    "save_onnx_model": False,
}

DEFAULT_CALIBRATION_CONFIG = {
    "dataset_name": "glue",
    "num_samples": 300,
    "dataset_config_name": "sst2",
    "dataset_split": "train",
    "preprocess_batch": True,
    "preprocess_class": "optimum_benchmark.preprocessors.glue.GluePreprocessor",
}
