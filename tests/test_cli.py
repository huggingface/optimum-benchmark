import os
import sys
from logging import getLogger
from pathlib import Path

import pytest

from optimum_benchmark.logging_utils import run_subprocess_and_log_stream_output

LOGGER = getLogger("test-cli")


FORCE_SEQUENTIAL = os.environ.get("FORCE_SEQUENTIAL", "0") == "1"
TEST_CONFIG_DIR = Path(__file__).parent / "configs"
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
    ]

    if not FORCE_SEQUENTIAL:
        args += [
            # to run the tests faster
            "hydra/launcher=joblib",
            "hydra.launcher.n_jobs=-1",
            "hydra.launcher.batch_size=1",
            "hydra.launcher.prefer=threads",
        ]

    if ROCR_VISIBLE_DEVICES is not None:
        args += [f'backend.device_ids="{ROCR_VISIBLE_DEVICES}"']
    elif CUDA_VISIBLE_DEVICES is not None:
        args += [f'backend.device_ids="{CUDA_VISIBLE_DEVICES}"']

    popen = run_subprocess_and_log_stream_output(LOGGER, args)
    assert popen.returncode == 0, f"Failed to run {config_name}"


@pytest.mark.parametrize("launcher", ["inline", "process", "torchrun"])
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
    if sys.platform == "win32":
        os.environ["USE_LIBUV"] = "0"

    args_1 = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        "_base_",
        "name=test",
        f"launcher={launcher}",
        # incompatible task and model to trigger an error
        "backend.task=image-classification",
        "backend.model=bert-base-uncased",
        "backend.device=cpu",
    ]

    popen_1 = run_subprocess_and_log_stream_output(LOGGER, args_1)
    assert popen_1.returncode == 1


@pytest.mark.parametrize("launcher", ["process", "torchrun"])
def test_cli_numactl(launcher):
    if sys.platform != "linux":
        pytest.skip("numactl is only supported on Linux")

    args = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        "_base_",
        "name=test",
        f"launcher={launcher}",
        "launcher.numactl=True",
        "backend.task=text-classification",
        "backend.model=bert-base-uncased",
        "backend.device=cpu",
    ]

    popen = run_subprocess_and_log_stream_output(LOGGER, args)
    assert popen.returncode == 0
