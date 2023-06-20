from typing import Dict, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
import shutil
import os
import gc

import torch
from torch import Tensor
from omegaconf import OmegaConf

from transformers import AutoConfig
from optimum.exporters import TasksManager
from transformers.utils.fx import symbolic_trace
from optimum.bettertransformer import BetterTransformer
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from src.backend.base import Backend, BackendConfig
from src.profiler.fx_profiler import FXProfilingWrapper

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

    no_weights: bool = False
    delete_cache: bool = False

    # inference options
    disable_grad: bool = "${is_inference:benchmark.name}"  # type: ignore
    eval_mode: bool = "${is_inference:benchmark.name}"  # type: ignore

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
        self.pretrained_config = AutoConfig.from_pretrained(
            self.model, **self.model_kwargs
        )
        self.automodel_class = TasksManager.get_model_class_for_task(
            task=self.task, framework="pt", model_type=self.pretrained_config.model_type
        )
        LOGGER.info(
            f"\t+ Infered AutoModel class {self.automodel_class.__name__} "
            f"for task {self.task} and model_type {self.pretrained_config.model_type}"
        )

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
        if config.no_weights:
            LOGGER.info(
                "\t+ Creating model with random weights on meta device")
            with init_empty_weights():
                self.pretrained_model = self.automodel_class.from_config(
                    self.pretrained_config,
                    torch_dtype=getattr(torch, config.torch_dtype),
                )

            LOGGER.info("\t+ Materializing the random weights model on cpu")
            self.pretrained_model = self.pretrained_model.to_empty(
                device="cpu")
            self.pretrained_model.tie_weights()

            if config.device_map is not None:
                LOGGER.info("\t+ Infering the device map")
                if config.device_map != "sequential":
                    max_memory = get_balanced_memory(
                        self.pretrained_model,
                        dtype=getattr(torch, config.torch_dtype),
                        low_zero=True,
                    )
                else:
                    max_memory = None

                device_map = infer_auto_device_map(
                    self.pretrained_model,
                    max_memory=max_memory,
                    dtype=getattr(torch, config.torch_dtype)
                )
                LOGGER.info("\t+ Dispatching the model to the device map")
                self.pretrained_model = dispatch_model(
                    self.pretrained_model, device_map=device_map)

        else:
            # load hosted weights model
            self.load_model(self.model, config)

        if config.device_map is not None:
            LOGGER.info("\t+ Resuling device map:")
            for k, v in self.pretrained_model.hf_device_map.items():
                LOGGER.info(f"\t\t+ {k} -> {v}")

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
            self.pretrained_model.forward = torch.compile(
                self.pretrained_model.forward)

        # Turn on autocast
        if config.autocast:
            LOGGER.info("\t+ Turning on autocast")
            self.autocast = True
        else:
            self.autocast = False

        # delete cache
        if config.delete_cache:
            LOGGER.info("\t+ Will delete cache after benchmarking")
            self.delete_cache = True
        else:
            self.delete_cache = False

    def load_model(self, pretrained_model_name_or_path: str, config: PyTorchConfig) -> None:
        """
        Model loading dispatcher for PyTorch backend
        """
        if config.quantization == "int8":
            LOGGER.info("\t+ Loading weights in int8 quantization")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                load_in_8bit=True,
                device_map=config.device_map,
                **self.model_kwargs,
            )
        elif config.quantization == "int4":
            LOGGER.info("\t+ Loading weights in int4 quantization")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                load_in_4bit=True,
                device_map=config.device_map,
                **self.model_kwargs,
            )
        else:
            LOGGER.info(f"\t+ Loading weights in {config.torch_dtype}")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                torch_dtype=getattr(torch, config.torch_dtype),
                device_map=config.device_map,
                **self.model_kwargs,
            )

    @contextmanager
    def amp_autocast(self):
        """
        Autocast context dispatcher for mixed precision.
        """
        if self.device == "cpu":
            with torch.cpu.amp.autocast(enabled=self.autocast):
                yield
        elif self.device == "cuda":
            with torch.cuda.amp.autocast(enabled=self.autocast):
                yield
        else:
            raise ValueError(f"Unknown device: {self.device}")

    def forward(self, input: Dict[str, Tensor]):
        with self.amp_autocast():
            output = self.pretrained_model(**input)

        return output

    def generate(self, input: Dict[str, Tensor], new_tokens: int) -> Tensor:
        with self.amp_autocast():
            output = self.pretrained_model.generate(  # type: ignore
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

    def delete_pretrained_model(self) -> None:
        del self.pretrained_model
        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def delete_model_hub_cache(self) -> None:
        model_cache_path = "models--" + self.model.replace("/", "--")
        model_cache_path = os.path.join(os.path.expanduser(
            "~/.cache/huggingface/hub"), model_cache_path)

        shutil.rmtree(model_cache_path, ignore_errors=True)

    def clean(self) -> None:
        LOGGER.info("Cleaning pytorch backend")
        self.delete_pretrained_model()

        if self.delete_cache:
            self.delete_model_hub_cache()
