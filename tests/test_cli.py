import os
from logging import getLogger

import pytest

from optimum_benchmark.logging_utils import run_subprocess_and_log_stream_output

LOGGER = getLogger("test-cli")


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
    ]

    popen = run_subprocess_and_log_stream_output(LOGGER, args)
    assert popen.returncode == 0, f"Failed to run {config_name}"


@pytest.mark.parametrize("launcher", ["inline", "process", "torchrun"])
def test_cli_exit_code(launcher):
    args_0 = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        "_base_",
        f"launcher={launcher}",
        "experiment_name=test",
        # compatible task and model
        "backend.task=text-classification",
        "backend.model=bert-base-uncased",
        "backend.device=cpu",
    ]

    popen_0 = run_subprocess_and_log_stream_output(LOGGER, args_0)
    assert popen_0.returncode == 0

    args_1 = [
        "optimum-benchmark",
        "--config-dir",
        TEST_CONFIG_DIR,
        "--config-name",
        "_base_",
        f"launcher={launcher}",
        "experiment_name=test",
        # incompatible task and model to trigger error
        "backend.task=image-classification",
        "backend.model=bert-base-uncased",
        "backend.device=cpu",
    ]

    popen_1 = run_subprocess_and_log_stream_output(LOGGER, args_1)
    assert popen_1.returncode == 1


# @pytest.mark.parametrize("launcher", ["inline", "process", "torchrun"])
# def test_cli_cuda_misc_device_isolation_error(launcher):
#     args_0 = [
#         "optimum-benchmark",
#         "--config-dir",
#         TEST_CONFIG_DIR,
#         "--config-name",
#         "_base_",
#         "--multirun",
#         f"launcher={launcher}",
#         "experiment_name=test",
#         # use cuda
#         "backend.device=cuda",
#         "backend.device_ids='0,1'",
#         # compatible task and model
#         "backend.task=text-classification",
#         "backend.model=bert-base-uncased,bert-base-uncased",
#         # set device isolation
#         "launcher.device_isolation=true",
#         "launcher.device_isolation_action=warn",
#         # run in parallel to trigger device isolation warning
#         "hydra/launcher=joblib",
#     ]

#     popen_0 = run_subprocess_and_log_stream_output(LOGGER, args_0)
#     assert popen_0.returncode == 0, "Expected to run successfully with device isolation warning"

#     args_1 = [
#         "optimum-benchmark",
#         "--config-dir",
#         TEST_CONFIG_DIR,
#         "--config-name",
#         "_base_",
#         "--multirun",
#         f"launcher={launcher}",
#         "experiment_name=test",
#         # use cuda
#         "backend.device=cuda",
#         "backend.device_ids='0,1'",
#         # compatible task and model
#         "backend.task=text-classification",
#         "backend.model=bert-base-uncased,bert-base-uncased",
#         # set device isolation
#         "launcher.device_isolation=true",
#         "launcher.device_isolation_action=error",
#         # run in parallel to trigger device isolation error
#         "hydra/launcher=joblib",
#     ]

#     popen_1 = run_subprocess_and_log_stream_output(LOGGER, args_1)
#     assert popen_1.returncode == 1, "Expected to fail due to device isolation error"
