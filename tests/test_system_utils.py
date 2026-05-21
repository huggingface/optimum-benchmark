import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

repo_root = Path(__file__).resolve().parents[1]
package = types.ModuleType("optimum_benchmark")
package.__path__ = [str(repo_root / "optimum_benchmark")]
sys.modules.setdefault("optimum_benchmark", package)

import_utils = types.ModuleType("optimum_benchmark.import_utils")
import_utils.is_amdsmi_available = lambda: False
import_utils.is_pynvml_available = lambda: False
import_utils.is_pyrsmi_available = lambda: False
sys.modules["optimum_benchmark.import_utils"] = import_utils

system_utils = importlib.import_module("optimum_benchmark.system_utils")


def test_get_cpu_on_darwin_uses_sysctl_argv():
    with (
        patch.object(system_utils.platform, "system", return_value="Darwin"),
        patch.object(system_utils.subprocess, "check_output", return_value="Apple M3 Max\n") as check_output,
    ):
        assert system_utils.get_cpu() == "Apple M3 Max"

    check_output.assert_called_once_with(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)


def test_get_cpu_on_linux_reads_proc_cpuinfo():
    cpuinfo = "processor\t: 0\nmodel name\t: Test CPU 123\n"

    with (
        patch.object(system_utils.platform, "system", return_value="Linux"),
        patch.object(system_utils.Path, "read_text", return_value=cpuinfo) as read_text,
    ):
        assert system_utils.get_cpu() == " Test CPU 123"

    read_text.assert_called_once_with(encoding="utf-8")


def test_command_succeeds_uses_argv():
    with patch.object(system_utils.subprocess, "call", return_value=0) as call:
        assert system_utils._command_succeeds("nvidia-smi") is True

    call.assert_called_once_with(
        ["nvidia-smi"],
        stdout=system_utils.subprocess.DEVNULL,
        stderr=system_utils.subprocess.DEVNULL,
    )


def test_command_succeeds_handles_missing_binary():
    with patch.object(system_utils.subprocess, "call", side_effect=FileNotFoundError):
        assert system_utils._command_succeeds("nvidia-smi") is False
