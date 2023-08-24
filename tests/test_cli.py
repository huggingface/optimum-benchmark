import os
import pytest
import subprocess
from omegaconf import OmegaConf


SINGLE_DEVICE_RUNS = [
    config
    for config in os.listdir("tests/configs")
    if config.endswith(".yaml")
    and config != "base_config.yaml"
    and "distributed" not in config
]

DISTRIBUTED_RUNS = [
    config
    for config in os.listdir("tests/configs")
    if config.endswith(".yaml")
    and config != "base_config.yaml"
    and "distributed" in config
]


@pytest.mark.parametrize("config_file", SINGLE_DEVICE_RUNS)
def test_single_device_runs(config_file):
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


@pytest.mark.parametrize("config_file", DISTRIBUTED_RUNS)
def test_distributed_runs(config_file):
    config_name = config_file.split(".")[0]

    env_set = OmegaConf.load(f"tests/configs/{config_file}")["hydra"]["job"]["env_set"]
    my_env = os.environ.copy()
    my_env.update(env_set)

    result = subprocess.run(
        [
            "optimum-benchmark",
            "--config-dir",
            "tests/configs",
            "--config-name",
            config_name,
        ],
        capture_output=True,
        env=my_env,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8")
