from typing import Dict, List, Optional
from dataclasses import dataclass
from logging import getLogger
import gc

import torch
from torch import Tensor
from omegaconf import OmegaConf
from transformers import AutoConfig
from optimum.exporters import TasksManager
from transformers.utils.fx import symbolic_trace
from optimum.bettertransformer import BetterTransformer

from src.backend.base import Backend, BackendConfig
from src.profiler.fx_profiler import FXProfilingWrapper
from src.utils import get_used_memory

# bachend logger
LOGGER = getLogger("pytorch")

# backend resolvers
OmegaConf.register_new_resolver(
    "is_inference", lambda benchmark: benchmark == "inference"
)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = "pytorch"
    version: str = torch.__version__
    _target_: str = "src.backend.pytorch.PyTorchBackend"

    # inference options
    disable_grad: bool = "${is_inference:benchmark.name}"
    eval_mode: bool = "${is_inference:benchmark.name}"

    # load options
    torch_dtype: str = "float32"  # "float32" or "float16"
    device_map: Optional[str] = None  # "auto"

    # quantization options
    quantization: Optional[str] = None  # "int8" or "int4"

    # optimization options
    bettertransformer: bool = False
    torch_compile: bool = False
    autocast: bool = False


class PyTorchBackend(Backend):
    def __init__(self, model: str, task: str, device: str, model_kwargs: dict):
        super().__init__(model, task, device, model_kwargs)

    def configure(self, config: PyTorchConfig) -> None:
        super().configure(config)

        # Torch specific environment variables
        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting pytorch inter_op_num_threads({config.inter_op_num_threads}))"
            )
            torch.set_num_threads(config.inter_op_num_threads)

        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting pytorch intra_op_num_threads({config.intra_op_num_threads}))"
            )
            torch.set_num_interop_threads(config.intra_op_num_threads)

        # Disable gradients
        if not config.disable_grad or config.eval_mode:
            LOGGER.info("\t+ Disabling gradients")
            torch.set_grad_enabled(False)

        model_type = AutoConfig.from_pretrained(
            self.model, **self.model_kwargs
        ).model_type
        LOGGER.info(f"\t+ Infered model type : {model_type}")

        self.automodel_class = TasksManager.get_model_class_for_task(
            task=self.task, model_type=model_type
        )
        LOGGER.info(f"\t+ Infered AutoModel class : {self.automodel_class.__name__}")

        # Load model
        if config.quantization == "int8":
            LOGGER.info("\t+ Loading weights in int8")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                load_in_8bit=True,
                device_map=config.device_map,
                **self.model_kwargs,
            )
        elif config.quantization == "int4":
            LOGGER.info("\t+ Loading weights in int4")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                load_in_4bit=True,
                device_map=config.device_map,
                **self.model_kwargs,
            )
        else:
            LOGGER.info(f"\t+ Loading weights in {config.torch_dtype}")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                torch_dtype=getattr(torch, config.torch_dtype),
                device_map=config.device_map,
                **self.model_kwargs,
            )

        # Move model to device
        if self.pretrained_model.device.type != self.device:
            LOGGER.info(f"\t+ Moving model to {self.device}")
            self.pretrained_model.to(self.device)

        # Turn on eval mode
        if config.eval_mode:
            LOGGER.info("\t+ Turning on eval mode")
            self.pretrained_model.eval()

        # Turn on better transformer inference
        if config.bettertransformer:
            LOGGER.info("\t+ Using optimum.bettertransformer")
            self.pretrained_model = BetterTransformer.transform(
                self.pretrained_model, keep_original_model=False
            )

        # Compile model
        if config.torch_compile:
            LOGGER.info("\t+ Using torch.compile")
            self.pretrained_model.forward = torch.compile(self.pretrained_model.forward)

        # Turn on autocast
        if config.autocast:
            LOGGER.info("\t+ Turning on autocast")
            self.autocast = True
        else:
            self.autocast = False

        LOGGER.debug(f"\t+ Device used memory: {get_used_memory(device=self.device)}")

    def forward(self, input: Dict[str, Tensor]):
        with torch.cuda.amp.autocast(enabled=self.autocast):
            output = self.pretrained_model(**input)
        return output

    def generate(self, input: Dict[str, Tensor], new_tokens: int) -> Tensor:
        with torch.cuda.amp.autocast(enabled=self.autocast):
            output = self.pretrained_model.generate(
                **input,
                pad_token_id=self.pretrained_model.config.eos_token_id,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                do_sample=False,
                use_cache=True,
                num_beams=1,
            )
        return output

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("\t+ Symbolic tracing model")
        self.pretrained_model = symbolic_trace(
            model=self.pretrained_model,
            input_names=input_names,
        )
        LOGGER.info("\t+ Wrapping model with profiler")
        self.pretrained_model = FXProfilingWrapper(self.pretrained_model)

    def clean(self) -> None:
        LOGGER.info("Cleaning onnxruntime backend")
        self._delete_pretrained_model()

    def _delete_pretrained_model(self) -> None:
        del self.pretrained_model
        gc.collect()
        torch.cuda.empty_cache()
