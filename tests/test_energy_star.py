import os
from logging import getLogger
from pathlib import Path

import pytest

from optimum_benchmark.logging_utils import run_subprocess_and_log_stream_output

LOGGER = getLogger("test-cli")


TEST_CONFIG_DIR = Path(__file__).parent.parent / "examples/energy_star"
TEST_CONFIG_NAMES = [
    config.split(".")[0]
    for config in os.listdir(TEST_CONFIG_DIR)
    if config.endswith(".yaml") and not (config.startswith("_") or config.endswith("_"))
]

ROCR_VISIBLE_DEVICES = os.environ.get("ROCR_VISIBLE_DEVICES", None)
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)


@pytest.mark.parametrize("config_name", TEST_CONFIG_NAMES)
def test_cli_configs(config_name):
    args = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        config_name,
        "scenario.num_samples=2",
        "scenario.input_shapes.batch_size=2",
    ]

    if ROCR_VISIBLE_DEVICES is not None:
        args += [f'backend.device_ids="{ROCR_VISIBLE_DEVICES}"']
    elif CUDA_VISIBLE_DEVICES is not None:
        args += [f'backend.device_ids="{CUDA_VISIBLE_DEVICES}"']

    popen = run_subprocess_and_log_stream_output(LOGGER, args)
    assert popen.returncode == 0, f"Failed to run {config_name}"
