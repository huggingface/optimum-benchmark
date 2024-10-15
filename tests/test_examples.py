import os
import subprocess
import sys
from logging import getLogger
from pathlib import Path

import pytest
import yaml

from optimum_benchmark.logging_utils import run_subprocess_and_log_stream_output

LOGGER = getLogger("test-example")

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
OUTPUT_DIR = Path(__file__).parent.parent / "runs"
YAML_CONFIGS = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".yaml") and f != "_base_.yaml"]
PYTHON_SCRIPTS = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".py")]

# can be run with pytest tests/test_example.py -s -k "cpu and ipex"
CPU_IPEX_CONFIGS = [
    "ipex_bert.yaml",
    "ipex_llama.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cpu and neural-compressor"
CPU_NEURAL_COMPRESSOR_CONFIGS = [
    "neural_compressor_ptq_bert.yaml",
    "numactl_bert.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cpu and onnxruntime"
CPU_ONNXRUNTIME_CONFIGS = [
    "onnxruntime_static_quant_vit.yaml",
    "onnxruntime_timm.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cpu and openvino"
CPU_OPENVINO_CONFIGS = [
    "openvino_diffusion.yaml",
    "openvino_static_quant_bert.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cpu and txi"
CPU_PY_TXI_CONFIGS = [
    "tei_bge.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cuda and pytorch"
CUDA_PYTORCH_CONFIGS = [
    "pytorch_bert.yaml",
    "pytorch_gpt2.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cuda and txi"
CUDA_PY_TXI_CONFIGS = [
    "tgi_llama.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cuda and tensorrt_llm"
CUDA_TENSORRT_LLM_CONFIGS = [
    # "trt_llama.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cuda and vllm"
CUDA_VLLM_CONFIGS = [
    "vllm_llama.yaml",
]

# can be run with pytest tests/test_example.py -s -k "mps and llama_cpp"
MPS_LLAMA_CPP_CONFIGS = [
    "llama_cpp_embedding.yaml",
    "llama_cpp_text_generation.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cuda and pytorch"
CUDA_PYTORCH_SCRIPTS = [
    "pytorch_bert.py",
    "pytorch_gpt2.py",
]

# Those tests are not run on the CI/CD pipeline as they are currently broken
UNTESTED_YAML_CONFIGS = [
    "energy_star.yaml",
    "trt_llama.yaml",
]

ALL_YAML_CONFIGS = (
    CUDA_PYTORCH_CONFIGS
    + CPU_IPEX_CONFIGS
    + MPS_LLAMA_CPP_CONFIGS
    + CPU_NEURAL_COMPRESSOR_CONFIGS
    + CPU_ONNXRUNTIME_CONFIGS
    + CPU_OPENVINO_CONFIGS
    + CPU_PY_TXI_CONFIGS
    + CUDA_PY_TXI_CONFIGS
    + CUDA_TENSORRT_LLM_CONFIGS
    + CUDA_VLLM_CONFIGS
    + UNTESTED_YAML_CONFIGS
)

ALL_PYTHON_SCRIPTS = CUDA_PYTORCH_SCRIPTS

assert set(ALL_YAML_CONFIGS) == set(YAML_CONFIGS), (
    f"Please add your new example config to the list of configs in test_example.py for it to be integrated in the CI/CD pipeline.\n"
    f"Difference between ALL_YAML_CONFIGS and YAML_CONFIGS:\n"
    f"In ALL_YAML_CONFIGS but not in YAML_CONFIGS: {set(ALL_YAML_CONFIGS) - set(YAML_CONFIGS)}\n"
    f"In YAML_CONFIGS but not in ALL_YAML_CONFIGS: {set(YAML_CONFIGS) - set(ALL_YAML_CONFIGS)}"
)

assert set(PYTHON_SCRIPTS) == set(ALL_PYTHON_SCRIPTS), (
    f"Please add your new example script to the list of scripts in test_example.py for it to be integrated in the CI/CD pipeline.\n"
    f"Difference between PYTHON_SCRIPTS and ALL_PYTHON_SCRIPTS:\n"
    f"In PYTHON_SCRIPTS but not in ALL_PYTHON_SCRIPTS: {set(PYTHON_SCRIPTS) - set(ALL_PYTHON_SCRIPTS)}\n"
    f"In ALL_PYTHON_SCRIPTS but not in PYTHON_SCRIPTS: {set(ALL_PYTHON_SCRIPTS) - set(PYTHON_SCRIPTS)}"
)


def extract_name_from_yaml(config_name):
    config_path = EXAMPLES_DIR / config_name

    with open(config_path, "r") as f:
        yaml_content = f.read()

    data = yaml.safe_load(yaml_content)

    return data.get("name")


def test_yaml_config(config_name):
    name = extract_name_from_yaml(config_name)

    args = [
        "optimum-benchmark",
        "--config-dir",
        str(EXAMPLES_DIR),
        "--config-name",
        config_name.split(".")[0],
    ]

    popen = run_subprocess_and_log_stream_output(LOGGER, args)
    assert popen.returncode == 0, f"Failed to run {config_name}"

    # Check if the benchmark produced any output
    output_dir = Path(OUTPUT_DIR) / name
    assert output_dir.exists(), f"No output directory found for {config_name}"

    # Check if there's at least one file in the output directory
    output_files = list(output_dir.glob("*"))
    assert len(output_files) > 0, f"No output files found for {config_name}"


def execute_python_script(script_name):
    script_path = EXAMPLES_DIR / script_name
    # Run the example file as a separate process
    process = subprocess.Popen(
        [sys.executable, str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Capture and display output in real-time
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())
            sys.stdout.flush()

    # Capture any remaining output
    stdout, stderr = process.communicate()

    # Create a result object similar to subprocess.run
    result = subprocess.CompletedProcess(
        args=[sys.executable, str(script_path)], returncode=process.returncode, stdout=stdout, stderr=stderr
    )

    # Check that the process completed successfully (return code 0)
    assert result.returncode == 0, f"Script {script_path} failed with error:\n{result.stderr}"

    # Check that there's no error output
    assert not result.stderr, f"Script {script_path} produced error output:\n{result.stderr}"


@pytest.mark.parametrize("config_name", CUDA_PYTORCH_CONFIGS)
def test_cuda_pytorch_configs(config_name):
    test_yaml_config(config_name)


@pytest.mark.parametrize("config_name", CPU_IPEX_CONFIGS)
def test_cpu_ipex_configs(config_name):
    test_yaml_config(config_name)


@pytest.mark.parametrize("config_name", MPS_LLAMA_CPP_CONFIGS)
def test_mps_llama_cpp_configs(config_name):
    test_yaml_config(config_name)


@pytest.mark.parametrize("config_name", CPU_NEURAL_COMPRESSOR_CONFIGS)
def test_cpu_neural_compressor_configs(config_name):
    test_yaml_config(config_name)


@pytest.mark.parametrize("config_name", CPU_ONNXRUNTIME_CONFIGS)
def test_cpu_onnxruntime_configs(config_name):
    test_yaml_config(config_name)


@pytest.mark.parametrize("config_name", CPU_OPENVINO_CONFIGS)
def test_cpu_openvino_configs(config_name):
    test_yaml_config(config_name)


@pytest.mark.parametrize("config_name", CPU_PY_TXI_CONFIGS)
def test_cpu_py_txi_configs(config_name):
    test_yaml_config(config_name)


@pytest.mark.parametrize("config_name", CUDA_PY_TXI_CONFIGS)
def test_cuda_py_txi_configs(config_name):
    test_yaml_config(config_name)


@pytest.mark.parametrize("config_name", CUDA_TENSORRT_LLM_CONFIGS)
def test_cuda_tensorrt_llm_configs(config_name):
    test_yaml_config(config_name)


@pytest.mark.parametrize("config_name", CUDA_VLLM_CONFIGS)
def test_cuda_vllm_configs(config_name):
    test_yaml_config(config_name)


@pytest.mark.parametrize("script_name", CUDA_PYTORCH_SCRIPTS)
def test_cuda_pytorch_scripts(script_name):
    execute_python_script(script_name)
