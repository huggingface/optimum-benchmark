import importlib.util
import os
import re
import subprocess

from setuptools import find_packages, setup

# Ensure we match the version set in src/optimum-benchmark/version.py
try:
    filepath = "optimum_benchmark/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

MIN_OPTIMUM_VERSION = "1.16.0"
INSTALL_REQUIRES = [
    # HF dependencies
    "transformers",
    "accelerate",
    "datasets",
    # Hydra
    "hydra-core",
    "omegaconf",
    # CPU
    "psutil",
    # Reporting
    "typing-extensions",
    "flatten_dict",
    "colorlog",
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

if USE_ROCM:
    AUTOAWQ = "autoawq@https://github.com/casper-hansen/AutoAWQ/releases/download/v0.2.1/autoawq-0.2.1+rocm571-cp310-cp310-linux_x86_64.whl"
    AUTOGPTQ = "auto-gptq@https://huggingface.github.io/autogptq-index/whl/rocm573/auto-gptq/auto_gptq-0.7.1%2Brocm5.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
else:
    AUTOAWQ = "autoawq==0.2.1"
    AUTOGPTQ = "auto-gptq==0.7.1"

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
    "llm-swarm": ["llm-swarm"],
    "py-txi": ["py-txi"],
    "vllm": ["vllm"],
    # optional dependencies
    "autoawq": [AUTOAWQ],
    "auto-gptq": ["optimum", AUTOGPTQ],
    "sentence-transformers": ["sentence-transformers"],
    "bitsandbytes": ["bitsandbytes"],
    "codecarbon": ["codecarbon"],
    "flash-attn": ["flash-attn"],
    "deepspeed": ["deepspeed"],
    "diffusers": ["diffusers"],
    "timm": ["timm"],
    "peft": ["peft"],
}


setup(
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={"console_scripts": ["optimum-benchmark=optimum_benchmark.cli:main"]},
    description="Optimum-Benchmark is a unified multi-backend utility for benchmarking "
    "Transformers, Timm, Diffusers and Sentence-Transformers with full support of "
    "Optimum's hardware optimizations & quantization schemes.",
    url="https://github.com/huggingface/optimum-benchmark",
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="benchmaek, transformers, quantization, pruning, optimization, training, inference, onnx, onnx runtime, intel, "
    "habana, graphcore, neural compressor, ipex, ipu, hpu, llm-swarm, py-txi, vllm, auto-gptq, autoawq, "
    "sentence-transformers, bitsandbytes, codecarbon, flash-attn, deepspeed, diffusers, timm, peft",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HuggingFace Inc. Special Ops Team",
    include_package_data=True,
    name="optimum-benchmark",
    version=__version__,
    license="Apache",
)
