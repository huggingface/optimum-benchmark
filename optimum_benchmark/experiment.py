import os
from logging import getLogger
from tempfile import TemporaryDirectory
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Type, Optional, Union, TYPE_CHECKING

from .system_utils import get_system_info
from .import_utils import get_hf_libs_info
from .benchmarks.report import BenchmarkReport
from .benchmarks.config import BenchmarkConfig
from .launchers.config import LauncherConfig
from .backends.config import BackendConfig

if TYPE_CHECKING:
    # avoid importing any torch to be able to set
    # the CUDA_VISIBLE_DEVICES environment variable
    # in BackendConfig __post_init__
    from .benchmarks.base import Benchmark
    from .launchers.base import Launcher
    from .backends.base import Backend

from json import dump
from flatten_dict import flatten
from hydra.utils import get_class
from transformers.configuration_utils import PushToHubMixin

LOGGER = getLogger("experiment")

EXPERIMENT_FILE_NAME = "experiment_config.json"


@dataclass
class ExperimentConfig(PushToHubMixin):
    # BACKEND CONFIGURATION
    backend: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # LAUNCHER CONFIGURATION
    launcher: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # BENCHMARK CONFIGURATION
    benchmark: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # Experiment name
    experiment_name: str

    task: Optional[str] = None  # deprecated
    model: Optional[str] = None  # deprecated
    device: Optional[str] = None  # deprecated
    library: Optional[str] = None  # deprecated

    # ENVIRONMENT CONFIGURATION
    environment: Dict = field(default_factory=lambda: {**get_system_info(), **get_hf_libs_info()})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        report_dict = self.to_dict()
        return flatten(report_dict, reducer="dot")

    def to_json(self, path: str, flat: bool = False) -> None:
        if flat:
            with open(path, "w") as f:
                dump(self.to_flat_dict(), f, indent=4)
        else:
            with open(path, "w") as f:
                dump(self.to_dict(), f, indent=4)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            kwargs["token"] = use_auth_token

        config_file_name = config_file_name if config_file_name is not None else EXPERIMENT_FILE_NAME

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        output_config_file = os.path.join(save_directory, config_file_name)
        self.to_json(output_config_file, flat=False)

        if push_to_hub:
            self._upload_modified_files(
                save_directory, repo_id, files_timestamps, commit_message=commit_message, token=kwargs.get("token")
            )


def run(benchmark_config: BenchmarkConfig, backend_config: BackendConfig) -> BenchmarkReport:
    try:
        # Allocate requested backend
        backend_factory: Type[Backend] = get_class(backend_config._target_)
        backend: Backend = backend_factory(backend_config)
    except Exception as e:
        LOGGER.error(f"Error during backend allocation: {e}")
        raise e

    try:
        # Allocate requested benchmark
        benchmark_factory: Type[Benchmark] = get_class(benchmark_config._target_)
        benchmark: Benchmark = benchmark_factory(benchmark_config)
    except Exception as e:
        LOGGER.error(f"Error during benchmark allocation: {e}")
        backend.clean()
        raise e

    try:
        # Benchmark the backend
        benchmark.run(backend)
        backend.clean()
    except Exception as e:
        LOGGER.error("Error during benchmark execution: %s", e)
        backend.clean()
        raise e

    try:
        report = benchmark.get_report()
    except Exception as e:
        LOGGER.error("Error during report generation: %s", e)
        raise e

    return report


def launch(experiment_config: ExperimentConfig) -> BenchmarkReport:
    # fix backend until deprecated model and device are removed
    if experiment_config.task is not None:
        LOGGER.warning("`task` is deprecated in experiment config. Use `backend.task` instead.")
        experiment_config.backend.task = experiment_config.task
    if experiment_config.model is not None:
        LOGGER.warning("`model` is deprecated in experiment config. Use `backend.model` instead.")
        experiment_config.backend.model = experiment_config.model
    if experiment_config.device is not None:
        LOGGER.warning("`device` is deprecated in experiment config. Use `backend.device` instead.")
        experiment_config.backend.device = experiment_config.device
    if experiment_config.library is not None:
        LOGGER.warning("`library` is deprecated in experiment config. Use `backend.library` instead.")
        experiment_config.backend.library = experiment_config.library

    original_dir = os.getcwd()
    tmpdir = TemporaryDirectory()

    if os.environ.get("BENCHMARK_INTERFACE", "API") == "API":
        # to not pollute the user's environment
        LOGGER.info("Launching experiment in a temporary directory.")
        os.chdir(tmpdir.name)

    launcher_config: LauncherConfig = experiment_config.launcher

    try:
        # Allocate requested launcher
        launcher_factory: Type[Launcher] = get_class(launcher_config._target_)
        launcher: Launcher = launcher_factory(launcher_config)
    except Exception as e:
        LOGGER.error(f"Error during launcher allocation: {e}")
        tmpdir.cleanup()
        raise e

    backend_config: BackendConfig = experiment_config.backend
    benchmark_config: BenchmarkConfig = experiment_config.benchmark

    try:
        output = launcher.launch(run, benchmark_config, backend_config)
    except Exception as e:
        LOGGER.error(f"Error during experiment launching: {e}")
        tmpdir.cleanup()
        raise e

    if os.environ.get("BENCHMARK_INTERFACE", "API") == "API":
        os.chdir(original_dir)
        tmpdir.cleanup()

    return output
