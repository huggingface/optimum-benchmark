import subprocess
from tempfile import TemporaryDirectory

CONFIG_DIR = "examples"


def test_pytorch_diffusion():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "pytorch_diffusion",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")


def test_onnxruntime_diffusion():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "onnxruntime_diffusion",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")
