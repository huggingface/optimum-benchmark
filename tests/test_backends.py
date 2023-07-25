import subprocess
from tempfile import TemporaryDirectory

CONFIG_DIR = "examples"


def test_pytorch():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "pytorch",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_onnxruntime():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "onnxruntime",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_openvino():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "openvino",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_neural_compressor():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "neural_compressor",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")
