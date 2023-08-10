from typing import Any, Dict, List, Optional
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from logging import getLogger
from datasets import Dataset
from torch import Tensor
import torch

from transformers.utils.fx import symbolic_trace
from optimum.bettertransformer import BetterTransformer

from optimum_benchmark.backends.base import Backend, BackendConfig
from optimum_benchmark.profilers.fx_profiler import FXProfilingWrapper


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
    _target_: str = "optimum_benchmark.backends.pytorch.PyTorchBackend"

    # load options
    no_weights: bool = False
    torch_dtype: Optional[str] = None

    # quantization options
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # optimization options
    bettertransformer: bool = False

    # compilation options
    torch_compile: bool = False
    torch_compile_config: DictConfig = DictConfig(
        {
            "fullgraph": False,
            "dynamic": False,
            "backend": "inductor",
            "mode": None,
            "options": None,
            "disable": False,
        }
    )
    # amp options
    amp_autocast: bool = False
    amp_dtype: Optional[str] = None

    # inference options
    disable_grad: bool = "${is_inference:${benchmark.name}}"  # type: ignore
    eval_mode: bool = "${is_inference:${benchmark.name}}"  # type: ignore


class PyTorchBackend(Backend):
    def __init__(self, model: str, task: str, device: str, hub_kwargs: DictConfig):
        super().__init__(model, task, device, hub_kwargs)

        LOGGER.info(
            f"\t+ Infered AutoModel class {self.automodel_class.__name__} "
            f"for task {self.task} and model_type {self.model_type}"
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
        if config.disable_grad:
            LOGGER.info("\t+ Disabling gradients")
            # everything that comes after this will have its gradients disabled
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
            self.load_model_from_pretrained(config)

        # Turn on eval mode
        if config.eval_mode and self.task not in [
            "stable-diffusion",
            "stable-diffusion-xl",
        ]:
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
            self.pretrained_model.forward = torch.compile(
                self.pretrained_model.forward,
                **config.torch_compile_config,
            )

        # pytorch autocast
        if config.amp_autocast:
            LOGGER.info(
                f"\t+ Enabling Automatic Mixed Precision with dtype: {self.amp_dtype}"
            )
        self.amp_autocast = config.amp_autocast
        self.amp_dtype = (
            getattr(torch, config.amp_dtype)  # in case of torch.dtype
            if config.amp_dtype is not None and hasattr(torch, config.amp_dtype)
            else None
        )

    def load_model_from_config(self, config: PyTorchConfig) -> None:
        LOGGER.info(
            f"\t+ Loading model from config in dtype : "
            f"{config.torch_dtype if config.torch_dtype is not None else 'default'} "
            "on meta device"
        )

        from accelerate import init_empty_weights
        from optimum_benchmark.backends.utils import (
            randomize_weights,
            quantize_dummy_model,
        )

        LOGGER.info("\t+ Initializing empty weights model on device: meta")
        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_config(
                config=self.pretrained_config,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.hub_kwargs.get("trust_remote_code", False),
            )

        if config.load_in_8bit or config.load_in_4bit:
            LOGGER.info("\t+ Materializing model on device: cpu")
            self.pretrained_model.to_empty(device="cpu")

            LOGGER.info("\t+ Randomizing model weights while on device: cpu")
            randomize_weights(self.pretrained_model)
            self.pretrained_model.tie_weights()

            from accelerate.utils import BnbQuantizationConfig

            bnb_quantization_config = BnbQuantizationConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                llm_int8_threshold=0,
                torch_dtype=self.torch_dtype,
                keep_in_fp32_modules=self.pretrained_model.keep_in_fp32_modules
                if hasattr(self.pretrained_model, "keep_in_fp32_modules")
                else None,
            )

            LOGGER.info("\t+ Quantizing model while on device: cpu")
            self.pretrained_model = quantize_dummy_model(
                model=self.pretrained_model,
                bnb_quantization_config=bnb_quantization_config,
            )

            LOGGER.info(f"\t+ Moving model to device: {self.device}")
            self.pretrained_model.to(self.device)
            self.pretrained_model.tie_weights()

        else:
            LOGGER.info(f"\t+ Materializing model on device: {self.device}")
            self.pretrained_model.to_empty(device=self.device)

            LOGGER.info("\t+ Randomizing model weights")
            randomize_weights(self.pretrained_model)
            self.pretrained_model.tie_weights()

    def load_model_from_pretrained(self, config: PyTorchConfig) -> None:
        LOGGER.info(
            f"\t+ Loading pretrained model weights in dtype: {config.torch_dtype} on device: {self.device}"
        )
        if self.task not in ["stable-diffusion", "stable-diffusion-xl"] and (
            config.load_in_8bit or config.load_in_4bit
        ):
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                load_in_8bit=config.load_in_8bit,
                load_in_4bit=config.load_in_4bit,
                llm_int8_threshold=0,
                **self.hub_kwargs,
            )
        elif self.task not in ["stable-diffusion", "stable-diffusion-xl"]:
            with self.device:
                self.pretrained_model = self.automodel_class.from_pretrained(
                    pretrained_model_name_or_path=self.model,
                    torch_dtype=self.torch_dtype,
                    **self.hub_kwargs,
                )
        else:
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                torch_dtype=self.torch_dtype,
                **self.hub_kwargs,
            )
            self.pretrained_model.to(self.device)

    def prepare_for_profiling(
        self,
        input_names: List[str],
        input_shapes: Dict[str, int],
    ) -> None:
        LOGGER.info("Preparing model for profiling")

        LOGGER.info("\t+ Symbolicly tracing model")
        self.pretrained_model = symbolic_trace(
            model=self.pretrained_model,
            input_names=input_names,
        )

        LOGGER.info("\t+ Wrapping model with FXProfilingWrapper")
        self.pretrained_model = FXProfilingWrapper(self.pretrained_model)

    def prepare_for_training(
        self,
        training_dataset: Dataset,
        training_arguments: Dict[str, Any],
    ) -> None:
        if self.device.type in ["cpu", "cuda"]:
            from transformers import Trainer, TrainingArguments

            LOGGER.info("\t+ Wrapping model with transformers.Trainer")
            training_arguments = TrainingArguments(**training_arguments)
            self.trainer = Trainer(
                model=self.pretrained_model,
                train_dataset=training_dataset,
                args=training_arguments,
            )
        elif self.device.type == "hpu":
            # habana example
            # from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
            # training_arguments = GaudiTrainingArguments(**training_arguments)
            # self.trainer = GaudiTrainer(
            #     model=self.pretrained_model,
            #     train_dataset=training_dataset,
            #     args=training_arguments,
            # )
            pass

    def forward(self, input: Dict[str, Tensor], **kwargs) -> Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_autocast,
        ):
            output = self.pretrained_model(**input, **kwargs)[0]

        return output

    def generate(self, input: Dict[str, Tensor], **kwargs) -> Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_autocast,
        ):
            output = self.pretrained_model.generate(**input, **kwargs)[0]

        return output

    def train(self) -> None:
        LOGGER.info("Training model")
        results = self.trainer.train()
        return results
