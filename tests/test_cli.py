import os
from logging import getLogger
from subprocess import PIPE, STDOUT, Popen

import pytest

LOGGER = getLogger("test")


CONFIG_NAMES = [
    config.split(".")[0]
    for config in os.listdir("tests/configs")
    if config.endswith(".yaml") and not (config.startswith("_") or config.endswith("_"))
]


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_config(config_name):
    args = [
        "optimum-benchmark",
        "--config-dir",
        "tests/configs",
        "--config-name",
        config_name,
        "--multirun",
    ]

    popen = Popen(args, stdout=PIPE, stderr=STDOUT)

    for line in iter(popen.stdout.readline, b""):
        if line is not None:
            LOGGER.info(line.decode("utf-8").rstrip())

    popen.wait()

    assert popen.returncode == 0, popen.stderr


def test_exit_code():
    args = [
        "optimum-benchmark",
        "--config-dir",
        "tests/configs",
        "--config-name",
        "cpu_inference_pytorch_bert_sweep",
        # incompatible task and model to trigger error
        "task=image-classification",
        "model=bert-base-uncased",
        # not using multirun to not sweep
    ]

    popen = Popen(args, stdout=PIPE, stderr=STDOUT)

    for line in iter(popen.stdout.readline, b""):
        if line is not None:
            LOGGER.info(line.decode("utf-8").rstrip())

    popen.wait()

    assert popen.returncode == 1, popen.stderr.decode("utf-8")
