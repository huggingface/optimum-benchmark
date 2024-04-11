import importlib.util
import os
import subprocess

from setuptools import find_packages, setup

OPTIMUM_BENCHMARK_VERSION = "0.2.0"

MIN_OPTIMUM_VERSION = "1.16.0"
INSTALL_REQUIRES = [
    # Mandatory HF dependencies
    "transformers",
    "accelerate",
    "datasets",
    # Hydra
    "hydra_colorlog",
    "hydra-core",
    "omegaconf",
    # CPU Memory
    "psutil",
    # Reporting
    "flatten_dict",
    "pandas",
]

try:
    subprocess.run(["nvidia-smi"], check=True)
    IS_NVIDIA_SYSTEM = True
except Exception:
    IS_NVIDIA_SYSTEM = False

try:
    subprocess.run(["rocm-smi"], check=True)
    IS_ROCM_SYSTEM = True
except Exception:
    IS_ROCM_SYSTEM = False

USE_CUDA = (os.environ.get("USE_CUDA", None) == "1") or IS_NVIDIA_SYSTEM
USE_ROCM = (os.environ.get("USE_ROCM", None) == "1") or IS_ROCM_SYSTEM

if USE_CUDA:
    INSTALL_REQUIRES.append("nvidia-ml-py")

if USE_ROCM:
    PYRSMI = "pyrsmi@git+https://github.com/ROCm/pyrsmi.git"
    INSTALL_REQUIRES.append(PYRSMI)
    if not importlib.util.find_spec("amdsmi"):
        print(
            "ROCm GPU detected without amdsmi installed. You won't be able to run process-specific VRAM tracking. "
            "Please install amdsmi from https://github.com/ROCm/amdsmi to enable this feature."
        )

AUTOGPTQ_CUDA = "auto-gptq==0.7.1"
AUTOGPTQ_ROCM = "auto-gptq@https://huggingface.github.io/autogptq-index/whl/rocm573/auto-gptq/auto_gptq-0.7.1%2Brocm5.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

AUTOAWQ_CUDA = "autoawq==0.2.1"
AUTOAWQ_ROCM = "autoawq@https://github.com/casper-hansen/AutoAWQ/releases/download/v0.2.1/autoawq-0.2.1+rocm571-cp310-cp310-linux_x86_64.whl"

EXTRAS_REQUIRE = {
    "quality": ["ruff"],
    "testing": ["pytest", "hydra-joblib-launcher"],
    # optimum backends
    "openvino": [f"optimum[openvino,nncf]>={MIN_OPTIMUM_VERSION}"],
    "onnxruntime": [f"optimum[onnxruntime]>={MIN_OPTIMUM_VERSION}"],
    "onnxruntime-gpu": [f"optimum[onnxruntime-gpu]>={MIN_OPTIMUM_VERSION}"],
    "neural-compressor": [f"optimum[neural-compressor]>={MIN_OPTIMUM_VERSION}"],
    "torch-ort": ["torch-ort", "onnxruntime-training", f"optimum>={MIN_OPTIMUM_VERSION}"],
    # other backends
    "llm-swarm": ["llm-swarm@git+https://github.com/huggingface/llm-swarm.git"],
    "py-txi": ["py-txi@git+https://github.com/IlyasMoutawwakil/py-txi.git"],
    # optional dependencies
    "autoawq": [AUTOAWQ_CUDA],
    "autoawq-rocm": [AUTOAWQ_ROCM],
    "auto-gptq": ["optimum", AUTOGPTQ_CUDA],
    "auto-gptq-rocm": ["optimum", AUTOGPTQ_ROCM],
    "bitsandbytes": ["bitsandbytes"],
    "codecarbon": ["codecarbon"],
    "deepspeed": ["deepspeed"],
    "diffusers": ["diffusers"],
    "timm": ["timm"],
    "peft": ["peft"],
}


setup(
    packages=find_packages(),
    name="optimum-benchmark",
    version=OPTIMUM_BENCHMARK_VERSION,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={"console_scripts": ["optimum-benchmark=optimum_benchmark.cli:benchmark_cli"]},
)
