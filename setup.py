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
PYRSMI = "pyrsmi@git+https://github.com/ROCm/pyrsmi.git"
if USE_ROCM:
    if not importlib.util.find_spec("amdsmi"):
        INSTALL_REQUIRES.append(PYRSMI)
else:
    try:
        subprocess.run(["rocm-smi"], stdout=subprocess.DEVNULL)
        if not importlib.util.find_spec("amdsmi"):
            INSTALL_REQUIRES.append(PYRSMI)
    except FileNotFoundError:
        pass

if PYRSMI in INSTALL_REQUIRES:
    print("ROCm GPU detected without amdsmi installed. Using pyrsmi instead but some features may not work.")


EXTRAS_REQUIRE = {
    "quality": ["ruff"],
    "testing": ["pytest", "hydra-joblib-launcher"],
    # optimum backends
    "openvino": [f"optimum[openvino,nncf]>={MIN_OPTIMUM_VERSION}"],
    "onnxruntime": [f"optimum[onnxruntime]>={MIN_OPTIMUM_VERSION}"],
    "neural-compressor": [f"optimum[neural-compressor]>={MIN_OPTIMUM_VERSION}"],
    "torch-ort": [f"optimum>={MIN_OPTIMUM_VERSION}", "onnxruntime-training", "torch-ort"],
    "onnxruntime-gpu": [f"optimum[onnxruntime-gpu]>={MIN_OPTIMUM_VERSION}"],
    # docker-based backends
    "py-tgi": ["py-tgi"],
    # third-party features
    "codecarbon": ["codecarbon"],
    "deepspeed": ["deepspeed"],
    "diffusers": ["diffusers"],
    "timm": ["timm"],
    "peft": ["peft"],
}


setup(
    name="optimum-benchmark",
    version=OPTIMUM_BENCHMARK_VERSION,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(),
    entry_points={"console_scripts": ["optimum-benchmark=optimum_benchmark.cli:benchmark_cli"]},
)
