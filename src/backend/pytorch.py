from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Optional
from dataclasses import dataclass
from logging import getLogger
from torch import Tensor
import torch

from transformers.utils.fx import symbolic_trace
from optimum.bettertransformer import BetterTransformer

from src.backend.base import Backend, BackendConfig
from src.profiler.fx_profiler import FXProfilingWrapper
from src.backend.utils import randomize_weights, quantize_model


# bachend logger
LOGGER = getLogger("pytorch")

# backend resolvers
OmegaConf.register_new_resolver(
    "is_inference", lambda benchmark_name: benchmark_name == "inference"
)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = "pytorch"
    version: str = torch.__version__
    _target_: str = "src.backend.pytorch.PyTorchBackend"

    # load options
    no_weights: bool = False
    torch_dtype: Optional[str] = None

    # quantization options
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # optimization options
    bettertransformer: bool = False
    torch_compile: bool = False
    amp_autocast: bool = False
    amp_dtype: Optional[str] = None

    # inference options
    disable_grad: bool = "${is_inference:${benchmark.name}}"  # type: ignore
    eval_mode: bool = "${is_inference:${benchmark.name}}"  # type: ignore


class PyTorchBackend(Backend):
    def __init__(self, model: str, task: str, device: str, cache_kwargs: DictConfig):
        super().__init__(model, task, device, cache_kwargs)

        LOGGER.info(
            f"\t+ Infered AutoModel class {self.automodel_class.__name__} "
            f"for task {self.task} and model_type {self.pretrained_config.model_type}"
        )

    def configure(self, config: PyTorchConfig) -> None:
        super().configure(config)

        # environment options
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
        if config.disable_grad or config.eval_mode:
            LOGGER.info("\t+ Disabling gradients")
            torch.set_grad_enabled(False)

        # Set torch dtype
        self.torch_dtype = (
            getattr(torch, config.torch_dtype)  # in case of torch.dtype
            if config.torch_dtype is not None and hasattr(torch, config.torch_dtype)
            else config.torch_dtype  # in case of string or None
        )

        # Load model
        if config.no_weights:
            self.load_model_from_config(config)
        else:
            # load hosted weights model
            self.load_model_from_pretrained(config)

        # Turn on eval mode
        if config.eval_mode:
            LOGGER.info("\t+ Turning on eval mode")
            self.pretrained_model.eval()

        # Turn on better transformer inference
        if config.bettertransformer:
            LOGGER.info("\t+ Using optimum.bettertransformer")
            self.pretrained_model = BetterTransformer.transform(  # type: ignore
                self.pretrained_model, keep_original_model=False
            )

        # Compile model
        if config.torch_compile:
            LOGGER.info("\t+ Using torch.compile on forward pass")
            self.pretrained_model.forward = torch.compile(self.pretrained_model.forward)
            if self.can_generate():
                LOGGER.info("\t+ Using torch.compile on generate pass")
                self.pretrained_model.generate = torch.compile(
                    self.pretrained_model.generate,
                    dynamic=True,
                )

        # pytorch autocast
        self.amp_autocast = config.amp_autocast
        self.amp_dtype = (
            getattr(torch, config.amp_dtype)  # in case of torch.dtype
            if config.amp_dtype is not None and hasattr(torch, config.amp_dtype)
            else None
        )
        if self.amp_autocast:
            LOGGER.info(
                f"\t+ Enabling Automatic Mixed Precision with dtype : "
                f"{self.amp_dtype if self.amp_dtype is not None else 'default'}"
            )

    def load_model_from_config(self, config: PyTorchConfig) -> None:
        LOGGER.info(
            f"\t+ Loading model from config in dtype : "
            f"{config.torch_dtype if config.torch_dtype is not None else 'default'} "
            "on meta device"
        )

        from accelerate import init_empty_weights

        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_config(
                config=self.pretrained_config,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.cache_kwargs.get("trust_remote_code", False),
            )

        if config.load_in_8bit or config.load_in_4bit:
            LOGGER.info(
                f"\t+ Quantizing random weights model to {'8bit' if config.load_in_8bit else '4bit'}"
                " using Accelerate's BitsAndBytes integration"
            )
            LOGGER.info("\t+ Materializing model on CPU")
            self.pretrained_model.to_empty(device="cpu")

            LOGGER.info("\t+ Randomizing model weights while on CPU")
            randomize_weights(self.pretrained_model)
            self.pretrained_model.tie_weights()

            from accelerate.utils import BnbQuantizationConfig

            bnb_quantization_config = BnbQuantizationConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                llm_int8_threshold=0 if config.load_in_8bit else 6,
                torch_dtype=self.torch_dtype,
                keep_in_fp32_modules=self.pretrained_model.keep_in_fp32_modules
                if hasattr(self.pretrained_model, "keep_in_fp32_modules")
                else None,
            )

            LOGGER.info("\t+ Quantizing model while on CPU")
            self.pretrained_model = quantize_model(
                model=self.pretrained_model,
                bnb_quantization_config=bnb_quantization_config,
            )

            LOGGER.info(f"\t+ Putting model on device {self.device}")
            self.pretrained_model.to(self.device)

        else:
            LOGGER.info(f"\t+ Materializing model on device {self.device}")
            self.pretrained_model.to_empty(device=self.device)

            LOGGER.info("\t+ Randomizing model weights")
            randomize_weights(self.pretrained_model)
            self.pretrained_model.tie_weights()

        # might be useful to make a helper method for this
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def load_model_from_pretrained(self, config: PyTorchConfig) -> None:
        LOGGER.info(
            f"\t+ Loading pretrained model weights in {config.torch_dtype} on {self.device} "
            f"with {'8bit' if config.load_in_8bit else '4bit' if config.load_in_4bit else 'no'} quantization"
        )
        self.pretrained_model = self.automodel_class.from_pretrained(
            pretrained_model_name_or_path=self.model,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            llm_int8_threshold=0 if config.load_in_8bit else 6,
            **self.cache_kwargs,
        )

    def forward(self, input: Dict[str, Tensor]):
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_autocast,
        ):
            output = self.pretrained_model(**input)[0]

        return output

    def generate(self, input: Dict[str, Tensor], new_tokens: int) -> Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_autocast,
        ):
            output = self.pretrained_model.generate(
                **input,
                pad_token_id=0,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                do_sample=False,
                use_cache=True,
                num_beams=1,
            )[0]

        return output

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing model for profiling")
        LOGGER.info("\t+ Symbolic tracing model")
        self.pretrained_model = symbolic_trace(  # type: ignore
            model=self.pretrained_model,
            input_names=input_names,
        )
        LOGGER.info("\t+ Wrapping model inside profiler")
        self.pretrained_model = FXProfilingWrapper(self.pretrained_model)
