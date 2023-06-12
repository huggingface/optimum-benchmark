from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List
import gc

import torch
from torch import Tensor
from transformers import AutoConfig
from optimum.exporters import TasksManager
from transformers.utils.fx import symbolic_trace
from optimum.bettertransformer import BetterTransformer

from src.backend.base import Backend, BackendConfig

from src.profiler.fx_profiler import FXProfilingWrapper

from src.utils import get_used_memory

BACKEND_NAME = "pytorch"
LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = torch.__version__
    _target_: str = "src.backend.pytorch.PyTorchBackend"

    # inference options
    disable_grad: bool = "${is_inference:benchmark.name}"
    eval_mode: bool = "${is_inference:benchmark.name}"

    # graph optimization options
    fp16: bool = False
    bettertransformer: bool = False
    torch_compile: bool = False


class PyTorchBackend(Backend):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)
        self.pretrained_config = AutoConfig.from_pretrained(self.model)

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

        # Load model
        automodel_class = TasksManager.get_model_class_for_task(
            task=self.task, model_type=self.pretrained_config.model_type
        )
        LOGGER.info(f"\t+ Loading {self.model} with {automodel_class.__name__}")
        self.pretrained_model = automodel_class.from_pretrained(self.model)

        # Move model to device
        if self.pretrained_model.device.type != self.device:
            LOGGER.info(f"\t+ Moving model to {self.device}")
            self.pretrained_model.to(self.device)

        LOGGER.debug(f"\t+ Device used memory: {get_used_memory(device=self.device)}")

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
            LOGGER.debug(
                f"\t+ Device used memory: {get_used_memory(device=self.device)}"
            )

        # Compile model
        if config.torch_compile:
            LOGGER.info("\t+ Using torch.compile")
            self.pretrained_model.forward = torch.compile(self.pretrained_model.forward)
            LOGGER.debug(
                f"\t+ Device used memory: {get_used_memory(device=self.device)}"
            )

        # Turn on fp16
        if config.fp16:
            LOGGER.info("\t+ Turning on fp16")
            self.fp16 = True
        else:
            self.fp16 = False

    def forward(self, input: Dict[str, Tensor]):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            output = self.pretrained_model(**input)
        return output

    def generate(self, input: Dict[str, Tensor], new_tokens: int) -> Tensor:
        with torch.cuda.amp.autocast(enabled=self.fp16):
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
