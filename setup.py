import os
import subprocess

from setuptools import find_packages, setup

OPTIMUM_VERSION = "1.14.0"

INSTALL_REQUIRES = [
    # Mandatory HF dependencies
    f"optimum>={OPTIMUM_VERSION}",  # backends, tasks and input generation
    "accelerate",  # distributed inference and no weights init
    # Hydra
    "hydra_colorlog",
    "hydra-core",
    "omegaconf",
    # Other
    "psutil",
    "pandas",
]

# We may allow to install CUDA or RoCm dependencies even when building in a non-CUDA or non-RoCm environment.
use_rocm = os.environ.get("USE_ROCM", None)
use_cuda = os.environ.get("USE_CUDA", None)

if use_cuda is None:
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL)
        use_cuda = "1"
    except FileNotFoundError:
        pass

if use_rocm is None:
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL)
        use_rocm = "1"
    except FileNotFoundError:
        pass

if use_cuda == "1":
    INSTALL_REQUIRES.append("py3nvml>=0.2.7")

if use_rocm == "1":
    INSTALL_REQUIRES.append("pyrsmi@git+https://github.com/RadeonOpenCompute/pyrsmi.git")

EXTRAS_REQUIRE = {
    "quality": ["black", "ruff"],
    "test": ["pytest", "hydra-joblib-launcher"],
    "report": ["matplotlib", "rich", "tabulate", "flatten_dict"],
    # cpu backends
    "openvino": [f"optimum[openvino,nncf]>={OPTIMUM_VERSION}"],
    "onnxruntime": [f"optimum[onnxruntime]>={OPTIMUM_VERSION}"],
    "neural-compressor": [f"optimum[neural-compressor]>={OPTIMUM_VERSION}"],
    # gpu backends
    "onnxruntime-gpu": [f"optimum[onnxruntime-gpu]>={OPTIMUM_VERSION}"],
    "onnxruntime-training": [
        "torch-ort",
        "onnxruntime-training",
        # # we use optimum from source, until the next release
        "optimum@git+https://github.com/huggingface/optimum.git",
    ],
    # docker-based backends
    "text-generation-inference": ["docker"],
    # specific settings
    "deepspeed": ["deepspeed"],
    "diffusers": ["diffusers"],
    "timm": ["timm"],
    "peft": ["peft"],
}


setup(
    name="optimum-benchmark",
    version="0.0.1",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "optimum-benchmark=optimum_benchmark.cli:benchmark_cli",
            "optimum-report=optimum_benchmark.cli:report_cli",
        ]
    },
)
