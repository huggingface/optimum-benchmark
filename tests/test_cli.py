import os
import subprocess
from logging import getLogger
from subprocess import PIPE, STDOUT, Popen

import pytest

LOGGER = getLogger("test")


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

    process = Popen(
        [
            "optimum-benchmark",
            "--config-dir",
            "tests/configs",
            "--config-name",
            config_name,
        ],
        stdout=PIPE,
        stderr=STDOUT,
    )

    for line in iter(process.stdout.readline, b""):
        if line is not None:
            LOGGER.info(line.decode("utf-8").rstrip())

    process.wait()

    assert process.returncode == 0, process.stderr


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
    )

    assert result.returncode == 1, result.stderr.decode("utf-8")
