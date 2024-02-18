import os
import subprocess
from setuptools import find_packages, setup

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
    "pandas",
    "flatten_dict",
]

# We may allow to install CUDA or RoCm dependencies even
# when building in a non-CUDA or non-ROCm environment.
USE_CUDA = os.environ.get("USE_CUDA", None) == "1"
USE_ROCM = os.environ.get("USE_ROCM", None) == "1"

if USE_CUDA:
    INSTALL_REQUIRES.append("nvidia-ml-py")
else:
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL)
        INSTALL_REQUIRES.append("nvidia-ml-py")
    except FileNotFoundError:
        pass

# we keep this as a check that amdsmi is installed since it's not available on pypi
if USE_ROCM:
    INSTALL_REQUIRES.append("amdsmi")
else:
    try:
        subprocess.run(["rocm-smi"], stdout=subprocess.DEVNULL)
        INSTALL_REQUIRES.append("amdsmi")
    except FileNotFoundError:
        pass


EXTRAS_REQUIRE = {
    "quality": ["ruff"],
    "testing": ["pytest", "hydra-joblib-launcher"],
    # api-based backends
    "openvino": [f"optimum[openvino,nncf]>={MIN_OPTIMUM_VERSION}"],
    "onnxruntime": [f"optimum[onnxruntime]>={MIN_OPTIMUM_VERSION}"],
    "neural-compressor": [f"optimum[neural-compressor]>={MIN_OPTIMUM_VERSION}"],
    "onnxruntime-gpu": [f"optimum[onnxruntime-gpu]>={MIN_OPTIMUM_VERSION}"],
    "torch-ort": [
        f"optimum>={MIN_OPTIMUM_VERSION}",
        "onnxruntime-training",
        "torch-ort",
    ],
    # docker-based backends
    "text-generation-inference": ["docker"],
    # specific settings
    "codecarbon": ["codecarbon"],
    "deepspeed": ["deepspeed"],
    "diffusers": ["diffusers"],
    "timm": ["timm"],
    "peft": ["peft"],
}


setup(
    name="optimum-benchmark",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(),
    version="0.1.0",
    entry_points={"console_scripts": ["optimum-benchmark=optimum_benchmark.cli:benchmark_cli"]},
)
