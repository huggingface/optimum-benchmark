import subprocess
from tempfile import TemporaryDirectory

CONFIG_DIR = "examples"


def test_pytorch_bert():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "pytorch_bert",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_onnxruntime_bert():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "onnxruntime_bert",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_openvino_bert():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "openvino_bert",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_neural_compressor_bert():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "neural_compressor_bert",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")
