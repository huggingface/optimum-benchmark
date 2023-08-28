import os
import subprocess

import pytest

SINGLE_DEVICE_RUNS = [
    config for config in os.listdir("tests/configs") if config.endswith(".yaml") and config != "base_config.yaml"
]


@pytest.mark.parametrize("config_file", SINGLE_DEVICE_RUNS)
def test_configs(config_file):
    config_name = config_file.split(".")[0]

    result = subprocess.run(
        [
            "optimum-benchmark",
            "--config-dir",
            "tests/configs",
            "--config-name",
            config_name,
            # "--multirun",
            # TODO: might be worth removing names from yaml configs and have a list of test models here
        ],
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8")
