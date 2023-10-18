import os

from setuptools import find_packages, setup

OPTIMUM_VERSION = "1.13.0"

INSTALL_REQUIRES = [
    # Mandatory HF dependencies
    f"optimum>={OPTIMUM_VERSION}",  # backends, tasks and input generation
    "accelerate",  # distributed inference and no weights init
    # Hydra
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.2",
    "hydra_colorlog>=1.2.0",
    "hydra-joblib-launcher>=1.2.0",
    # Other
    "psutil>=5.9.0",
    "pandas>=2.0.0",
]

use_rocm = os.environ.get("USE_ROCM", None)
ues_cuda = os.environ.get("USE_CUDA", None)
if ues_cuda == "1":
    INSTALL_REQUIRES.append("py3nvml>=0.2.7")

if use_rocm == "1":
    INSTALL_REQUIRES.append("pyrsmi@git+https://github.com/RadeonOpenCompute/pyrsmi.git")

EXTRAS_REQUIRE = {
    "test": ["pytest"],
    "quality": ["black", "ruff"],
    "report": ["flatten_dict", "matplotlib", "seaborn", "rich"],
    # cpu backends
    "openvino": [f"optimum[openvino,nncf]>={OPTIMUM_VERSION}"],
    "onnxruntime": [f"optimum[onnxruntime]>={OPTIMUM_VERSION}", "torch==2.0.1"],
    "onnxruntime-gpu": [f"optimum[onnxruntime-gpu]>={OPTIMUM_VERSION}", "torch==2.0.1"],
    "neural-compressor": [f"optimum[neural-compressor]>={OPTIMUM_VERSION}"],
    # server-like backend
    "text-generation-inference": ["docker>=6.1.3"],
    # specific settings
    "diffusers": ["diffusers"],
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
            "optimum-benchmark=optimum_benchmark.experiment:run_experiment",
            "optimum-report=optimum_benchmark.report:generate_report",
        ]
    },
)
