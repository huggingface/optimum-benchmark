from typing import Any, Dict


DEFAULT_OPTIMIZATION_CONFIG = {
    "optimization_level": 1,  # 0, 1, 2, 99
    "optimize_for_gpu": "${is_gpu:${device}}",
    "fp16": False,
    "enable_transformers_specific_optimizations": True,
    "enable_gelu_approximation": False,
    "disable_gelu_fusion": False,
    "disable_layer_norm_fusion": False,
    "disable_attention_fusion": False,
    "disable_skip_layer_norm_fusion": True,
    "disable_bias_skip_layer_norm_fusion": False,
    "disable_bias_gelu_fusion": False,
    "use_mask_index": False,
    "no_attention_mask": False,
    "disable_embed_layer_norm_fusion": True,
    "disable_shape_inference": False,
    "use_multi_head_attention": False,
    "enable_gemm_fast_gelu_fusion": False,
    "use_raw_attention_mask": False,
    "disable_group_norm_fusion": True,
    "disable_packed_kv": True,
}

DEFAULT_QUANTIZATION_CONFIG = {
    "is_static": False,
    "format": "QOperator",  # QOperator, QDQ
    "mode": "IntegerOps",  # QLinearOps, IntegerOps
    "activations_dtype": "QUInt8",  # QInt8, QUInt8
    "activations_symmetric": False,
    "weights_dtype": "QInt8",  # QInt8, QUInt8
    "weights_symmetric": True,
    "per_channel": False,
    "reduce_range": False,
    "operators_to_quantize": [
        "MatMul",
        "Add",
    ],
}

DEFAULT_CALIBRATION_CONFIG = {
    "dataset_name": "glue",
    "num_samples": 300,
    "dataset_config_name": "sst2",
    "dataset_split": "train",
    "preprocess_batch": True,
    "preprocess_class": "optimum_benchmark.preprocessors.glue.GluePreprocessor",
}


def infer_device_id(device: str) -> int:
    """
    Infer the device id from the given device string.
    """

    import torch

    if device == "cuda":
        return torch.cuda.current_device()
    elif torch.device(device).type == "cuda":
        return torch.device(device).index
    elif torch.device(device).type == "cpu":
        return -1
    else:
        raise ValueError(f"Unknown device '{device}'")


def format_ort_quantization_dict(quantization_dict: Dict[str, Any]) -> None:
    """
    Format the quantization dictionary for onnxruntime.
    """

    from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType

    if quantization_dict.get("format", None) is not None:
        quantization_dict["format"] = QuantFormat.from_string(
            quantization_dict["format"]
        )
    if quantization_dict.get("mode", None) is not None:
        quantization_dict["mode"] = QuantizationMode.from_string(
            quantization_dict["mode"]
        )
    if quantization_dict.get("activations_dtype", None) is not None:
        quantization_dict["activations_dtype"] = QuantType.from_string(
            quantization_dict["activations_dtype"]
        )
    if quantization_dict.get("weights_dtype", None) is not None:
        quantization_dict["weights_dtype"] = QuantType.from_string(
            quantization_dict["weights_dtype"]
        )

    return quantization_dict
