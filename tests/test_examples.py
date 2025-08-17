import os
from logging import getLogger
from pathlib import Path

import pytest

from optimum_benchmark.logging_utils import run_subprocess_and_log_stream_output

LOGGER = getLogger("test-examples")

os.environ["TRANSFORMERS_IS_CI"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEST_CONFIG_DIR = Path(__file__).parent.parent / "examples"
TEST_CONFIG_NAMES = [
    config.split(".")[0]
    for config in os.listdir(TEST_CONFIG_DIR)
    if config.endswith(".yaml") and not (config.startswith("_") or config.endswith("_"))
]
TEST_SCRIPT_PATHS = [
    str(TEST_CONFIG_DIR / filename) for filename in os.listdir(TEST_CONFIG_DIR) if filename.endswith(".py")
]

ROCR_VISIBLE_DEVICES = os.environ.get("ROCR_VISIBLE_DEVICES", None)
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)


@pytest.mark.parametrize("config_name", TEST_CONFIG_NAMES)
def test_cli_configs(config_name):
    if config_name == "cpu_ipex_bert":
        model = "hf-internal-testing/tiny-random-BertModel"
    elif config_name == "cpu_ipex_llama":
        model = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    elif config_name == "cpu_llama_cpp_text_generation":
        model = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    elif config_name == "cpu_llama_cpp_embedding":
        model = "nomic-ai/nomic-embed-text-v1.5-GGUF"
    elif config_name == "cpu_onnxruntime_static_quant_vit":
        model = "hf-internal-testing/tiny-random-ViTModel"
    elif config_name == "cpu_openvino_8bit_bert":
        model = "hf-internal-testing/tiny-random-BertModel"
    elif config_name == "cpu_openvino_diffusion":
        model = "hf-internal-testing/tiny-stable-diffusion-torch"
    elif config_name == "cuda_pytorch_bert":
        model = "hf-internal-testing/tiny-random-BertModel"
    elif config_name.startswith("cuda_pytorch_llama"):
        model = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    elif config_name == "cuda_pytorch_vlm":
        model = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
    elif config_name in ["cuda_tgi_llama", "cuda_trt_llama", "cuda_vllm_llama"]:
        model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    else:
        raise ValueError(f"Unsupported config name: {config_name}")

    args = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        config_name,
        "scenario.warmup_runs=1",
        "scenario.input_shapes.batch_size=1",
        "++scenario.input_shapes.sequence_length=16",
        "++scenario.generate_kwargs.max_new_tokens=16",
        "++scenario.generate_kwargs.min_new_tokens=16",
        "++scenario.call_kwargs.num_inference_steps=4",
        "backend.model=" + model,
        "++backend.reshape_kwargs.batch_size=1",
        "++backend.reshape_kwargs.sequence_length=16",
    ]

    if ROCR_VISIBLE_DEVICES is not None:
        args += [f'backend.device_ids="{ROCR_VISIBLE_DEVICES}"']
    elif CUDA_VISIBLE_DEVICES is not None:
        args += [f'backend.device_ids="{CUDA_VISIBLE_DEVICES}"']

    popen = run_subprocess_and_log_stream_output(LOGGER, args)
    assert popen.returncode == 0, f"Failed to run {config_name}"


@pytest.mark.parametrize("script_path", TEST_SCRIPT_PATHS)
def test_api_scripts(script_path):
    args = ["python", script_path]

    popen = run_subprocess_and_log_stream_output(LOGGER, args)
    assert popen.returncode == 0, f"Failed to run {script_path}"
