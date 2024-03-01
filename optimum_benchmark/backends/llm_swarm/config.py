from dataclasses import dataclass
from typing import Optional

from ...import_utils import llm_swarm_version
from ..config import BackendConfig


@dataclass
class LLMSwarmConfig(BackendConfig):
    name: str = "llm-swarm"
    version: Optional[str] = llm_swarm_version()
    _target_: str = "optimum_benchmark.backends.llm_swarm.backend.LLMSwarmBackend"

    # optimum benchmark specific
    no_weights: bool = False

    # llm-swarm specific
    gpus: int = 8
    instances: int = 1
    inference_engine: str = "tgi"
    volume: str = "/fsx/ilyas/.cache"
    per_instance_max_parallel_requests: int = 500
    slurm_template_path: str = "/fsx/ilyas/swarm-templates/tgi_h100.template.slurm"
    load_balancer_template_path: str = "/fsx/ilyas/swarm-templates/nginx.template.conf"
    debug_endpoint: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

        # so that downloaded artifacts are stored in the same place
        self.hub_kwargs["cache_dir"] = self.volume
