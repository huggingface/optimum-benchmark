DEFAULT_QUANTIZATION_CONFIG = {
    "device": "cpu",
    "backend": "default",
    "domain": "auto",
    "recipes": {},
    "quant_format": "default",
    "inputs": [],
    "outputs": [],
    "approach": "static",
    "calibration_sampling_size": [100],
    "op_type_dict": None,
    "op_name_dict": None,
    "reduce_range": None,
    "example_inputs": None,
    "excluded_precisions": [],
    "quant_level": "auto",
    "accuracy_criterion": {
        "higher_is_better": True,
        "criterion": "relative",
        "tolerable_loss": 0.01,
    },
    "tuning_criterion": {
        "strategy": "basic",
        "strategy_kwargs": None,
        "timeout": 0,
        "max_trials": 100,
        "objective": "performance",
    },
    "diagnosis": False,
}

DEFAULT_CALIBRATION_CONFIG = {
    "dataset_name": "glue",
    "num_samples": 300,
    "dataset_config_name": "sst2",
    "dataset_split": "train",
    "preprocess_batch": True,
    "preprocess_class": "optimum_benchmark.preprocessors.glue.GluePreprocessor",
}
