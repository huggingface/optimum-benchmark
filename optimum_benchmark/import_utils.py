import importlib.util

_torch_available = importlib.util.find_spec("torch") is not None
_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
_is_openvino_available = importlib.util.find_spec("openvino") is not None
_is_neural_compressor_available = importlib.util.find_spec("neural-compressor") is not None

def is_torch_available():
    return _torch_available

def is_onnxruntime_available():
    return _onnxruntime_available

def is_openvino_available():
    return _is_openvino_available

def is_neural_compressor_available():
    return _is_neural_compressor_available