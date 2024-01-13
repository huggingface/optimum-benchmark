import importlib.metadata
import importlib.util

_transformers_available = importlib.util.find_spec("transformers") is not None
_accelerate_available = importlib.util.find_spec("accelerate") is not None
_diffusers_available = importlib.util.find_spec("diffusers") is not None
_optimum_available = importlib.util.find_spec("optimum") is not None
_torch_available = importlib.util.find_spec("torch") is not None
_onnx_available = importlib.util.find_spec("onnx") is not None
_tensorrt_available = importlib.util.find_spec("tensorrt") is not None
_peft_available = importlib.util.find_spec("peft") is not None
_py3nvml_available = importlib.util.find_spec("py3nvml") is not None
_torch_distributed_available = importlib.util.find_spec("torch.distributed") is not None
_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
_openvino_available = importlib.util.find_spec("openvino") is not None
_neural_compressor_available = importlib.util.find_spec("neural_compressor") is not None
_pyrsmi_available = importlib.util.find_spec("pyrsmi") is not None
_codecarbon_available = importlib.util.find_spec("codecarbon") is not None
_amdsmi_available = importlib.util.find_spec("amdsmi") is not None
_tensorflow_available = importlib.util.find_spec("tensorflow") is not None
_timm_available = importlib.util.find_spec("timm") is not None


def is_timm_available():
    return _timm_available


def is_tensorflow_available():
    return _tensorflow_available


def is_tensorrt_available():
    return _tensorrt_available


def is_peft_available():
    return _peft_available


def is_onnx_available():
    return _onnx_available


def is_optimum_available():
    return _optimum_available


def is_onnxruntime_available():
    return _onnxruntime_available


def is_py3nvml_available():
    return _py3nvml_available


def is_pyrsmi_available():
    return _pyrsmi_available


def is_amdsmi_available():
    return _amdsmi_available


def is_torch_available():
    return _torch_available


def is_torch_distributed_available():
    return _torch_distributed_available


def is_codecarbon_available():
    return _codecarbon_available


def torch_version():
    if is_torch_available():
        return importlib.metadata.version("torch")


def tesnorrt_version():
    if is_tensorrt_available():
        return importlib.metadata.version("tensorrt")


def onnxruntime_version():
    try:
        return "ort:" + importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        try:
            return "ort-gpu:" + importlib.metadata.version("onnxruntime-gpu")
        except importlib.metadata.PackageNotFoundError:
            try:
                return "ort-training:" + importlib.metadata.version("onnxruntime-training")
            except importlib.metadata.PackageNotFoundError:
                return "ort:unknown"


def openvino_version():
    if _openvino_available:
        return importlib.metadata.version("openvino")


def neural_compressor_version():
    if _neural_compressor_available:
        return importlib.metadata.version("neural_compressor")


def optimum_version():
    if _optimum_available:
        return importlib.metadata.version("optimum")


def transformers_version():
    if _transformers_available:
        return importlib.metadata.version("transformers")


def accelerate_version():
    if _accelerate_available:
        return importlib.metadata.version("accelerate")


def diffusers_version():
    if _diffusers_available:
        return importlib.metadata.version("diffusers")
