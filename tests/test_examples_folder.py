import os
import subprocess
import sys
from logging import getLogger
from pathlib import Path

import pytest

from optimum_benchmark.logging_utils import run_subprocess_and_log_stream_output

LOGGER = getLogger("test-example")

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
OUTPUT_DIR = Path(__file__).parent.parent / "runs"
EXAMPLE_CONFIGS = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".yaml") and f != "_base_.yaml"]

# can be run with pytest tests/test_example.py -s -k "cpu and pytorch"
CPU_PYTORCH_EXAMPLE_CONFIGS = [
    "pytorch_bert.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cpu and ipex"
CPU_IPEX_EXAMPLE_CONFIGS = [
    "ipex_bert.yaml",
    "ipex_llama.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cpu and neural-compressor"
CPU_NEURAL_COMPRESSOR_EXAMPLE_CONFIGS = [
    "neural_compressor_ptq_bert.yaml",
    "numactl_bert.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cpu and onnxruntime"
CPU_ONNXRUNTIME_EXAMPLE_CONFIGS = [
    "onnxruntime_static_quant_vit.yaml",
    "onnxruntime_timm.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cpu and openvino"
CPU_OPENVINO_EXAMPLE_CONFIGS = [
    "openvino_diffusion.yaml",
    "openvino_static_quant_bert.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cpu and txi"
CPU_TXI_EXAMPLE_CONFIGS = [
    "tei_bge.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cuda and pytorch"
CUDA_PYTORCH_EXAMPLE_CONFIGS = [
    "energy_star.yaml",
    "pytorch_bert.yaml",
    "pytorch_llama.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cuda and txi"
CUDA_TXI_EXAMPLE_CONFIGS = [
    "tgi_llama.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cuda and trt"
CUDA_TRT_EXAMPLE_CONFIGS = [
    "trt_llama.yaml",
]

# can be run with pytest tests/test_example.py -s -k "cuda and vllm"
CUDA_VLLM_EXAMPLE_CONFIGS = [
    "vllm_llama.yaml",
]

# can be run with pytest tests/test_example.py -s -k "mps and llama_cpp"
MPS_LLAMA_CPP_EXAMPLE_CONFIGS = [
    "llama_cpp_embedding.yaml",
    "llama_cpp_text_generation.yaml",
]

ALL_CONFIGS = (
    CUDA_PYTORCH_EXAMPLE_CONFIGS
    + CPU_PYTORCH_EXAMPLE_CONFIGS
    + CPU_IPEX_EXAMPLE_CONFIGS
    + MPS_LLAMA_CPP_EXAMPLE_CONFIGS
    + CPU_NEURAL_COMPRESSOR_EXAMPLE_CONFIGS
    + CPU_ONNXRUNTIME_EXAMPLE_CONFIGS
    + CPU_OPENVINO_EXAMPLE_CONFIGS
    + CPU_TXI_EXAMPLE_CONFIGS
    + CUDA_TXI_EXAMPLE_CONFIGS
    + CUDA_TRT_EXAMPLE_CONFIGS
    + CUDA_VLLM_EXAMPLE_CONFIGS
)

assert set(ALL_CONFIGS) == set(EXAMPLE_CONFIGS), (
    f"Please add your new example config to the list of configs in test_example.py for it to be integrated in the CI/CD pipeline.\n"
    f"Difference between ALL_CONFIGS and EXAMPLE_CONFIGS:\n"
    f"In ALL_CONFIGS but not in EXAMPLE_CONFIGS: {set(ALL_CONFIGS) - set(EXAMPLE_CONFIGS)}\n"
    f"In EXAMPLE_CONFIGS but not in ALL_CONFIGS: {set(EXAMPLE_CONFIGS) - set(ALL_CONFIGS)}"
)


def test_example_configs(config_name):
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
    output_dir = Path(OUTPUT_DIR) / config_name.split(".")[0]
    assert output_dir.exists(), f"No output directory found for {config_name}"

    # Check if there's at least one file in the output directory
    output_files = list(output_dir.glob("*"))
    assert len(output_files) > 0, f"No output files found for {config_name}"


@pytest.mark.cuda
@pytest.mark.pytorch
@pytest.mark.parametrize("config_name", CUDA_PYTORCH_EXAMPLE_CONFIGS)
def test_cuda_pytorch_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.cpu
@pytest.mark.pytorch
@pytest.mark.parametrize("config_name", CPU_PYTORCH_EXAMPLE_CONFIGS)
def test_cpu_pytorch_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.cpu
@pytest.mark.ipex
@pytest.mark.parametrize("config_name", CPU_IPEX_EXAMPLE_CONFIGS)
def test_cpu_ipex_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.mps
@pytest.mark.llama_cpp
@pytest.mark.parametrize("config_name", MPS_LLAMA_CPP_EXAMPLE_CONFIGS)
def test_mps_llama_cpp_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.cpu
@pytest.mark.neural_compressor
@pytest.mark.parametrize("config_name", CPU_NEURAL_COMPRESSOR_EXAMPLE_CONFIGS)
def test_cpu_neural_compressor_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.cpu
@pytest.mark.onnxruntime
@pytest.mark.parametrize("config_name", CPU_ONNXRUNTIME_EXAMPLE_CONFIGS)
def test_cpu_onnxruntime_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.cpu
@pytest.mark.openvino
@pytest.mark.parametrize("config_name", CPU_OPENVINO_EXAMPLE_CONFIGS)
def test_cpu_openvino_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.cpu
@pytest.mark.txi
@pytest.mark.parametrize("config_name", CPU_TXI_EXAMPLE_CONFIGS)
def test_cpu_txi_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.cuda
@pytest.mark.txi
@pytest.mark.parametrize("config_name", CUDA_TXI_EXAMPLE_CONFIGS)
def test_cuda_txi_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.cuda
@pytest.mark.tensorrt_llm
@pytest.mark.parametrize("config_name", CUDA_TRT_EXAMPLE_CONFIGS)
def test_cuda_trt_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.cuda
@pytest.mark.vllm
@pytest.mark.parametrize("config_name", CUDA_VLLM_EXAMPLE_CONFIGS)
def test_cuda_vllm_examples(config_name):
    test_example_configs(config_name)


@pytest.mark.cuda
@pytest.mark.pytorch
@pytest.mark.parametrize(
    "example_file",
    [
        "examples/pytorch_bert.py",
        "examples/pytorch_llama.py",
    ],
)
def test_example_file_execution(example_file):
    # Run the example file as a separate process
    result = subprocess.run([sys.executable, example_file], capture_output=True, text=True)

    # Check that the process completed successfully (return code 0)
    assert result.returncode == 0, f"Example {example_file} failed with error:\n{result.stderr}"

    # Check that there's no error output
    assert not result.stderr, f"Example {example_file} produced error output:\n{result.stderr}"
