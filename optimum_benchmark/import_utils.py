import importlib.metadata
import importlib.util
from pathlib import Path
from subprocess import STDOUT, check_output
from typing import Optional

_transformers_available = importlib.util.find_spec("transformers") is not None
_accelerate_available = importlib.util.find_spec("accelerate") is not None
_diffusers_available = importlib.util.find_spec("diffusers") is not None
_optimum_available = importlib.util.find_spec("optimum") is not None
_torch_available = importlib.util.find_spec("torch") is not None
_onnx_available = importlib.util.find_spec("onnx") is not None
_tensorrt_available = importlib.util.find_spec("tensorrt") is not None
_peft_available = importlib.util.find_spec("peft") is not None
_pynvml_available = importlib.util.find_spec("pynvml") is not None
_torch_distributed_available = importlib.util.find_spec("torch.distributed") is not None
_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
_openvino_available = importlib.util.find_spec("openvino") is not None
_neural_compressor_available = importlib.util.find_spec("neural_compressor") is not None
_codecarbon_available = importlib.util.find_spec("codecarbon") is not None
_amdsmi_available = importlib.util.find_spec("amdsmi") is not None
_tensorflow_available = importlib.util.find_spec("tensorflow") is not None
_timm_available = importlib.util.find_spec("timm") is not None
_diffusers_available = importlib.util.find_spec("diffusers") is not None
_torch_ort_available = importlib.util.find_spec("torch_ort") is not None
_deepspeed_available = importlib.util.find_spec("deepspeed") is not None
_tensorrt_llm_available = importlib.util.find_spec("tensorrt_llm") is not None
_psutil_available = importlib.util.find_spec("psutil") is not None
_optimum_benchmark_available = importlib.util.find_spec("optimum_benchmark") is not None
_py_txi_available = importlib.util.find_spec("py_txi") is not None
_pyrsmi_available = importlib.util.find_spec("pyrsmi") is not None
_llm_swarm_available = importlib.util.find_spec("llm_swarm") is not None
_zentorch_available = importlib.util.find_spec("zentorch") is not None
_vllm_available = importlib.util.find_spec("vllm") is not None


def is_vllm_available():
    return _vllm_available


def is_zentorch_available():
    return _zentorch_available


def is_llm_swarm_available():
    return _llm_swarm_available


def is_pyrsmi_available():
    return _pyrsmi_available


def is_py_txi_available():
    return _py_txi_available


def is_psutil_available():
    return _psutil_available


def is_transformers_available():
    return _transformers_available


def is_tensorrt_llm_available():
    return _tensorrt_llm_available


def is_deepspeed_available():
    return _deepspeed_available


def is_torch_ort_available():
    return _torch_ort_available


def is_accelerate_available():
    return _accelerate_available


def is_diffusers_available():
    return _diffusers_available


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


def is_pynvml_available():
    return _pynvml_available


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
                return None


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


def torch_ort_version():
    if _torch_ort_available:
        return importlib.metadata.version("torch_ort")


def timm_version():
    if _timm_available:
        return importlib.metadata.version("timm")


def peft_version():
    if _peft_available:
        return importlib.metadata.version("peft")


def tesnorrt_llm_version():
    if _tensorrt_llm_available:
        return importlib.metadata.version("tensorrt_llm")


def optimum_benchmark_version():
    if _optimum_benchmark_available:
        return importlib.metadata.version("optimum_benchmark")


def py_txi_version():
    if _py_txi_available:
        return importlib.metadata.version("py_txi")


def llm_swarm_version():
    if _llm_swarm_available:
        return importlib.metadata.version("llm_swarm")


def vllm_version():
    if _vllm_available:
        return importlib.metadata.version("vllm")


def get_git_revision_hash(package_name: str) -> Optional[str]:
    """
    Returns the git commit SHA of a package installed from a git repository.
    """

    try:
        path = Path(importlib.util.find_spec(package_name).origin).parent
    except Exception:
        return None

    try:
        git_hash = check_output(["git", "rev-parse", "HEAD"], cwd=path, stderr=STDOUT).strip().decode("utf-8")

    except Exception:
        return None

    return git_hash


def get_hf_libs_info():
    return {
        "optimum_benchmark_version": optimum_benchmark_version(),
        "optimum_benchmark_commit": get_git_revision_hash("optimum_benchmark"),
        "transformers_version": transformers_version() if is_transformers_available() else None,
        "transformers_commit": get_git_revision_hash("transformers"),
        "accelerate_version": accelerate_version() if is_accelerate_available else None,
        "accelerate_commit": get_git_revision_hash("accelerate"),
        "diffusers_version": diffusers_version() if is_diffusers_available() else None,
        "diffusers_commit": get_git_revision_hash("diffusers"),
        "optimum_version": optimum_version() if is_optimum_available() else None,
        "optimum_commit": get_git_revision_hash("optimum"),
        "timm_version": timm_version() if is_timm_available() else None,
        "timm_commit": get_git_revision_hash("timm"),
        "peft_version": peft_version() if is_peft_available() else None,
        "peft_commit": get_git_revision_hash("peft"),
    }
