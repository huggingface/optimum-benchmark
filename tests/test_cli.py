import os
import sys
from logging import getLogger
from pathlib import Path

import mock
import pytest

from optimum_benchmark.logging_utils import run_subprocess_and_log_stream_output

LOGGER = getLogger("test-cli")

os.environ["TRANSFORMERS_IS_CI"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    if launcher == "torchrun" and sys.platform == "win32":
        pytest.skip("torchrun is not supported on Windows")

    args_0 = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        "_base_",
        "name=test",
        "launcher=" + launcher,
        # compatible task and model
        "backend.model=hf-internal-testing/tiny-random-BertModel",
        "backend.task=text-classification",
        "backend.device=cpu",
        # input shapes
        "+scenario.input_shapes.batch_size=1",
        "+scenario.input_shapes.sequence_length=16",
    ]

    popen_0 = run_subprocess_and_log_stream_output(LOGGER, args_0)
    assert popen_0.returncode == 0


@pytest.mark.parametrize("launcher", ["inline", "process", "torchrun"])
def test_cli_exit_code_1(launcher):
    if launcher == "torchrun" and sys.platform == "win32":
        pytest.skip("torchrun is not supported on Windows")

    args_1 = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        "_base_",
        "name=misc_test",
        "launcher=" + launcher,
        # incompatible task and model to trigger an error
        "backend.model=hf-internal-testing/tiny-random-BertModel",
        "backend.task=image-classification",
        "backend.device=cpu",
        # input shapes
        "+scenario.input_shapes.batch_size=1",
        "+scenario.input_shapes.sequence_length=16",
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
        "name=misc_test",
        "launcher=" + launcher,
        "launcher.numactl=True",
        "backend.model=hf-internal-testing/tiny-random-BertModel",
        "backend.task=text-classification",
        "backend.device=cpu",
        # input shapes
        "+scenario.input_shapes.sequence_length=16",
        "+scenario.input_shapes.batch_size=1",
    ]

    popen = run_subprocess_and_log_stream_output(LOGGER, args)
    assert popen.returncode == 0


@pytest.mark.parametrize("launcher", ["process", "torchrun"])
@mock.patch.dict(os.environ, {"FILE_BASED_COMM_THRESHOLD": "1"})
def test_cli_file_based_comm(launcher):
    if launcher == "torchrun" and sys.platform == "win32":
        pytest.skip("torchrun is not supported on Windows")

    args = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        "_base_",
        "name=misc_test",
        "launcher=" + launcher,
        # compatible task and model
        "backend.model=hf-internal-testing/tiny-random-BertModel",
        "backend.task=text-classification",
        "backend.device=cpu",
        # input shapes
        "+scenario.input_shapes.sequence_length=16",
        "+scenario.input_shapes.batch_size=1",
    ]

    popen = run_subprocess_and_log_stream_output(LOGGER, args)
    assert popen.returncode == 0
