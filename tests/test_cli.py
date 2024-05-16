import os
from logging import getLogger

import pytest

from optimum_benchmark.logging_utils import run_subprocess_and_log_stream_output

LOGGER = getLogger("test")

TEST_CONFIG_DIR = "/".join(__file__.split("/")[:-1] + ["configs"])
TEST_CONFIG_NAMES = [
    config.split(".")[0]
    for config in os.listdir(TEST_CONFIG_DIR)
    if config.endswith(".yaml") and not (config.startswith("_") or config.endswith("_"))
]


@pytest.mark.parametrize("config_name", TEST_CONFIG_NAMES)
def test_cli_configs(config_name):
    args = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        config_name,
        # to run the tests faster (comment for debugging)
        "hydra/launcher=joblib",
        "hydra.launcher.batch_size=1",
        "hydra.launcher.prefer=threads",
    ]

    popen = run_subprocess_and_log_stream_output(LOGGER, args)
    assert popen.returncode == 0, f"Failed to run {config_name}"


@pytest.mark.parametrize("launcher", ["inline", "process"])
def test_cli_exit_code_0(launcher):
    args_0 = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        "_base_",
        "name=test",
        f"launcher={launcher}",
        # compatible task and model
        "backend.task=text-classification",
        "backend.model=bert-base-uncased",
        "backend.device=cpu",
    ]

    popen_0 = run_subprocess_and_log_stream_output(LOGGER, args_0)
    assert popen_0.returncode == 0


@pytest.mark.parametrize("launcher", ["inline", "process", "torchrun"])
def test_cli_exit_code_1(launcher):
    args_1 = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        "_base_",
        "name=test",
        f"launcher={launcher}",
        # incompatible task and model to trigger error
        "backend.task=image-classification",
        "backend.model=bert-base-uncased",
        "backend.device=cpu",
    ]

    popen_1 = run_subprocess_and_log_stream_output(LOGGER, args_1)
    assert popen_1.returncode == 1
