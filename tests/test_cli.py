import os
import subprocess

import pytest

SINGLERUNS = [
    config
    for config in os.listdir("tests/configs")
    if config.endswith(".yaml")
    and not config.startswith("multirun")
    and not (config.startswith("_") or config.endswith("_"))
]

MULTIRUNS = [
    config
    for config in os.listdir("tests/configs")
    if config.endswith(".yaml")
    and config.startswith("multirun")
    and not (config.startswith("_") or config.endswith("_"))
]


@pytest.mark.parametrize("config_file", SINGLERUNS)
def test_single_run(config_file):
    config_name = config_file.split(".")[0]

    result = subprocess.run(
        [
            "optimum-benchmark",
            "--config-dir",
            "tests/configs",
            "--config-name",
            config_name,
        ],
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_exit_code():
    result = subprocess.run(
        [
            "optimum-benchmark",
            "--config-dir",
            "tests/configs",
            "--config-name",
            "cpu_pytorch_inference_bert",
            "model=inexistent_model",
        ],
        capture_output=True,
    )

    assert result.returncode == 1, result.stderr.decode("utf-8")
