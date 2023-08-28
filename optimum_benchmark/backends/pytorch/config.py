import importlib.metadata
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from ..base import BackendConfig

OmegaConf.register_new_resolver(
    "device_count",
    lambda: len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")),
)
OmegaConf.register_new_resolver(
    "is_inference",
    lambda benchmark_name: benchmark_name == "inference",
)
OmegaConf.register_new_resolver(
    "pytorch_version",
    lambda: importlib.metadata.version("torch"),
)

DEVICE_MAPS = ["auto", "sequential"]
AMP_DTYPES = ["bfloat16", "float16"]
TORCH_DTYPES = ["bfloat16", "float16", "float32", "auto"]

GPTQ_CONFIG = {
    "bits": 4,
}
BNB_CONFIG = {
    "load_in_8bit": False,
    "load_in_4bit": False,
    "llm_int8_threshold": 0.0,
}
QUANTIZATION_CONFIGS = {
    "gptq": GPTQ_CONFIG,
    "bnb": BNB_CONFIG,
}
COMPILE_CONFIG = {
    "fullgraph": False,
    "dynamic": False,
    "backend": "inductor",
    "mode": None,
    "options": None,
    "disable": False,
}
# from launchConfig in https://github.com/pytorch/pytorch/blob/v2.0.0/torch/distributed/launcher/api.py#L29 adjusted
# to defaults of torch.distributed.run in https://github.com/pytorch/pytorch/blob/v2.0.0/torch/distributed/run.py#L770
DDP_CONFIG = {
    "min_nodes": 1,
    "max_nodes": 1,
    "run_id": "none",
    "nproc_per_node": "${device_count:}",
    "role": "default",
    "rdzv_endpoint": "127.0.0.1:29500",
    "rdzv_backend": "static",
    "rdzv_configs": {
        "timeout": 900,
        "rank": 0,
    },
    "max_restarts": 0,
    "monitor_interval": 5,
    "start_method": "spawn",
    "log_dir": None,
    "metrics_cfg": {},
    "local_addr": None,
}


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = "pytorch"
    version: str = "${pytorch_version:}"
    _target_: str = "optimum_benchmark.backends.pytorch.backned.PyTorchBackend"

    # load options
    no_weights: bool = False
    device_map: Optional[str] = None
    torch_dtype: Optional[str] = None

    # inference options
    disable_grad: bool = "${is_inference:${benchmark.name}}"
    eval_mode: bool = "${is_inference:${benchmark.name}}"

    # automatic mixed precision options
    amp_autocast: bool = False
    amp_dtype: Optional[str] = None

    # compilation options
    torch_compile: bool = False
    torch_compile_config: Dict[str, Any] = field(default_factory=dict)

    # optimization options
    bettertransformer: bool = False

    # quantization options
    quantization_strategy: Optional[str] = None
    quantization_config: Dict[str, Any] = field(default_factory=dict)

    # training options
    use_ddp: bool = False
    ddp_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        if self.torch_compile:
            self.torch_compile_config = OmegaConf.to_container(
                OmegaConf.merge(COMPILE_CONFIG, self.torch_compile_config)
            )

        if self.device_map is not None:
            assert CUDA_VISIBLE_DEVICES is not None, "`device_map` can only be used when CUDA_VISIBLE_DEVICES is set."

            if self.device_map not in DEVICE_MAPS:
                raise ValueError(f"`device_map` must be one of {DEVICE_MAPS}. Got {self.device_map} instead.")

        if self.torch_dtype is not None:
            if self.torch_dtype not in TORCH_DTYPES:
                raise ValueError(f"`torch_dtype` must be one of {TORCH_DTYPES}. Got {self.torch_dtype} instead.")

        if self.amp_dtype is not None:
            if self.amp_dtype not in AMP_DTYPES:
                raise ValueError(f"`amp_dtype` must be one of {AMP_DTYPES}. Got {self.amp_dtype} instead.")

        if self.quantization_strategy is not None:
            if self.quantization_strategy not in QUANTIZATION_CONFIGS:
                raise ValueError(
                    f"`quantization_strategy` must be one of {list(QUANTIZATION_CONFIGS.keys())}. Got {self.quantization_strategy} instead."
                )
            QUANTIZATION_CONFIG = QUANTIZATION_CONFIGS[self.quantization_strategy]
            self.quantization_config = OmegaConf.to_container(
                OmegaConf.merge(QUANTIZATION_CONFIG, self.quantization_config)
            )

        if self.use_ddp:
            if CUDA_VISIBLE_DEVICES is None:
                raise ValueError("`use_ddp` can only be used when CUDA_VISIBLE_DEVICES is set.")

            self.ddp_config = OmegaConf.to_container(OmegaConf.merge(DDP_CONFIG, self.ddp_config), resolve=True)
            # TODO: check if it's not possible to use DDP with multiple nodes
            if self.ddp_config["max_nodes"] > 1 or self.ddp_config["min_nodes"] > 1:
                raise NotImplementedError("Currently, PyTorch DDP benchmark only supports training on a single node.")
