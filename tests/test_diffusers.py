import subprocess
from tempfile import TemporaryDirectory

CONFIG_DIR = "examples"


def test_diffusers():
    with TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "optimum-benchmark",
                "--config-dir",
                CONFIG_DIR,
                "--config-name",
                "diffusers",
                f"hydra.run.dir={tmpdir}",
            ],
            capture_output=True,
        )

    assert result.returncode == 0, result.stderr.decode("utf-8")
