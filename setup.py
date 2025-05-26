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

MIN_OPTIMUM_VERSION = "1.18.0"
INSTALL_REQUIRES = [
    # HF dependencies
    "huggingface-hub",
    "transformers",
    "accelerate",
    "datasets",
    "hf_xet",
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
    "rich",
]

try:
    subprocess.run(["nvidia-smi"], check=True)
    IS_NVIDIA_SYSTEM = True
except Exception:
    IS_NVIDIA_SYSTEM = False

try:
    subprocess.run(["amd-smi"], check=True)
    IS_ROCM_SYSTEM = True
except Exception:
    IS_ROCM_SYSTEM = False

USE_CUDA = (os.environ.get("USE_CUDA", None) == "1") or IS_NVIDIA_SYSTEM
USE_ROCM = (os.environ.get("USE_ROCM", None) == "1") or IS_ROCM_SYSTEM

if USE_CUDA:
    INSTALL_REQUIRES.append("nvidia-ml-py")

if USE_ROCM:
    INSTALL_REQUIRES.extend(["pyrsmi", "amdsmi"])

DEV_REQUIRE = ["ruff", "pytest", "mock", "hydra-joblib-launcher"]

EXTRAS_REQUIRE = {
    "quality": ["ruff"],
    "dev": DEV_REQUIRE,
    "tests": DEV_REQUIRE,
    "testing": DEV_REQUIRE,
    # optimum backends
    "ipex": [f"optimum[ipex]>={MIN_OPTIMUM_VERSION}"],
    "tensorrt-llm": [f"optimum[nvidia]>={MIN_OPTIMUM_VERSION}"],
    "openvino": [f"optimum[openvino,nncf]>={MIN_OPTIMUM_VERSION}"],
    "onnxruntime": [f"optimum[onnxruntime]>={MIN_OPTIMUM_VERSION}"],
    "onnxruntime-gpu": [f"optimum[onnxruntime-gpu]>={MIN_OPTIMUM_VERSION}"],
    # other backends
    "llama-cpp": ["llama-cpp-python"],
    "llm-swarm": ["llm-swarm"],
    "py-txi": ["py-txi"],
    "vllm": ["vllm"],
    # optional dependencies
    "sentence-transformers": ["sentence-transformers"],
    "gptqmodel": ["gptqmodel", "optimum"],
    "codecarbon": ["codecarbon<3.0.0"],
    "bitsandbytes": ["bitsandbytes"],
    "deepspeed": ["deepspeed<0.16"],
    "flash-attn": ["flash-attn"],
    "diffusers": ["diffusers"],
    "torchao": ["torchao"],
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
    keywords="benchmark, transformers, quantization, pruning, optimization, training, inference, onnx, onnx runtime, intel, "
    "habana, graphcore, neural compressor, ipex, ipu, hpu, llm-swarm, py-txi, vllm, llama-cpp, gptqmodel, "
    "sentence-transformers, bitsandbytes, codecarbon, flash-attn, deepspeed, diffusers, timm, peft",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HuggingFace Inc. Special Ops Team",
    include_package_data=True,
    name="optimum-benchmark",
    version=__version__,
    license="Apache",
)
