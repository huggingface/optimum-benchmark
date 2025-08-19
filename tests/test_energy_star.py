import os
from logging import getLogger
from pathlib import Path

import pytest

from optimum_benchmark.logging_utils import run_subprocess_and_log_stream_output

LOGGER = getLogger("test-cli")

os.environ["TRANSFORMERS_IS_CI"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEST_CONFIG_DIR = Path(__file__).parent.parent / "energy_star"
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
    if config_name == "automatic_speech_recognition":
        model = "optimum-internal-testing/tiny-random-whisper"
    elif config_name == "image_classification":
        model = "hf-internal-testing/tiny-random-ViTModel"
    elif config_name == "image_to_text":
        model = "hf-internal-testing/tiny-random-BlipModel"
    elif config_name == "object_detection":
        model = "hf-internal-testing/tiny-random-DetrModel"
    elif config_name == "question_answering":
        model = "hf-internal-testing/tiny-random-BertModel"
    elif config_name == "sentence_similarity":
        model = "hf-internal-testing/tiny-random-BertModel"
    elif config_name == "text_classification":
        model = "hf-internal-testing/tiny-random-BertModel"
    elif config_name == "summarization":
        model = "hf-internal-testing/tiny-random-BartModel"
    elif config_name == "t5_question_answering":
        model = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
    elif config_name == "t5_summarization":
        model = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
    elif config_name == "t5_text_classification":
        model = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
    elif config_name == "t5_text_generation":
        model = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
    elif config_name == "text_to_image":
        model = "hf-internal-testing/tiny-stable-diffusion-torch"
    elif config_name == "text_generation":
        model = "tiny-random/gpt-oss"
    else:
        raise ValueError(f"Unknown config name: {config_name}")

    args = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR.as_posix(),
        "--config-name",
        config_name,
        "backend.device=cpu",
        "scenario.energy=true",
        "scenario.memory=true",
        "scenario.latency=true",
        "scenario.num_samples=1",
        "scenario.warmup_runs=1",
        "scenario.input_shapes.batch_size=1",
        "++scenario.generate_kwargs.max_new_tokens=16",
        "++scenario.generate_kwargs.min_new_tokens=16",
        "++scenario.call_kwargs.num_inference_steps=4",
        "launcher.device_isolation=false",
        "backend.device_map=null",
        f"backend.model={model}",
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
