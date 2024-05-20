from dataclasses import dataclass, field
from typing import Any, Dict

from ..hub_utils import PushToHubMixin, classproperty
from ..import_utils import get_hf_libs_info
from ..system_utils import get_system_info


@dataclass
class BenchmarkConfig(PushToHubMixin):
    name: str

    # BACKEND CONFIGURATION
    backend: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # SCENARIO CONFIGURATION
    scenario: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # LAUNCHER CONFIGURATION
    launcher: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # ENVIRONMENT CONFIGURATION
    environment: Dict[str, Any] = field(default_factory=lambda: {**get_system_info(), **get_hf_libs_info()})

    @classproperty
    def default_filename(cls) -> str:
        return "benchmark_config.json"
