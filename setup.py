from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    # Mandatory HF dependencies
    "optimum>=1.13.0",  # backends, tasks and input generation
    "transformers>=4.20.0",  # pytorch, models, configs, hub, etc.
    "accelerate>=0.22.0",  # distributed inference and no weights init
    # Hydra
    "omegaconf==2.3.0",
    "hydra-core==1.3.2",
    "hydra_colorlog==1.2.0",
    "hydra-joblib-launcher==1.2.0",
    # Other
    "codecarbon==2.3.1",
    "psutil==5.9.0",
    "pandas>=2.0.0",
]


EXTRAS_REQUIRE = {
    "test": ["pytest"],
    "quality": ["black", "ruff"],
    "report": ["flatten_dict", "matplotlib", "seaborn", "rich"],
    # cpu backends
    "openvino": ["optimum[openvino,nncf]"],
    "onnxruntime": ["optimum[onnxruntime]"],
    "neural-compressor": ["optimum[neural-compressor]"],
    # cuda backends
    "onnxruntime-gpu": ["py3nvml", "optimum[onnxruntime-gpu]"],
    "text-generation-inference": ["py3nvml-0.2.7", "docker==6.1.3"],
    # specific settings
    "peft": ["peft"],
    "diffusers": ["diffusers"],
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
