from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from logging import getLogger
import os
import gc


import torch
from torch import Tensor
from accelerate import init_empty_weights
from omegaconf import DictConfig, OmegaConf
from torch import __version__ as torch_version
from transformers.utils.fx import symbolic_trace
from transformers import Trainer, TrainingArguments
from optimum.bettertransformer import BetterTransformer
from transformers import BitsAndBytesConfig, GPTQConfig
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import elastic_launch, LaunchConfig


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.utils import ModelOutput
    from transformers import TrainerState, TrainerCallback


from .base import Backend, BackendConfig
from ..profilers.fx_profiler import FXProfilingWrapper
from .utils.pytorch_utils import (
    DEFAULT_COMPILE_CONFIG,
    DEFAULT_DDP_CONFIG,
    randomize_weights,
    get_worker_logger,
)


# bachend logger
LOGGER = getLogger("pytorch")

# backend resolvers
OmegaConf.register_new_resolver(
    "is_inference", lambda benchmark_name: benchmark_name == "inference"
)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = "pytorch"
    version: str = torch_version
    _target_: str = "optimum_benchmark.backends.pytorch.PyTorchBackend"

    # load options
    no_weights: bool = False
    device_map: Optional[str] = None
    torch_dtype: Optional[str] = None

    # quantization options
    quantization_strategy: Optional[str] = None
    quantization_config: Optional[Dict[str, Any]] = None

    # optimization options
    bettertransformer: bool = False

    # compilation options
    torch_compile: bool = False
    torch_compile_kwargs: Optional[Dict] = None

    # amp options
    amp_autocast: bool = False
    amp_dtype: Optional[str] = None

    # inference options
    disable_grad: bool = "${is_inference:${benchmark.name}}"  # type: ignore
    eval_mode: bool = "${is_inference:${benchmark.name}}"  # type: ignore

    # training options
    use_ddp: bool = False
    ddp_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """
        Here we perform checks and transformations on the config.
        But we never modify the types of the config values.
        """

        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        if self.torch_compile:
            self.torch_compile_kwargs = OmegaConf.merge(
                self.torch_compile_kwargs
                if self.torch_compile_kwargs is not None
                else {},
                DEFAULT_COMPILE_CONFIG,
            )

        if self.device_map is not None:
            assert self.device_map in ["auto", "sequential"], (
                "`device_map` must be one of ['auto', 'sequential']. "
                "are supported in Optimum-Bnechmark. "
                f"Got {type(self.device_map)} instead."
            )
            assert (
                CUDA_VISIBLE_DEVICES is not None
            ), "`device_map` can only be used when CUDA_VISIBLE_DEVICES is set."

        if self.torch_dtype is not None:
            assert self.torch_dtype in ["bfloat16", "float16", "float32", "auto"], (
                "`torch_dtype` must be one of ['bfloat16', 'float16', 'float32', "
                f"'auto']. Got {self.torch_dtype} instead."
            )

        if self.amp_dtype is not None:
            assert self.amp_dtype in ["bfloat16", "float16", "float32"], (
                "`amp_dtype` must be one of ['bfloat16', 'float16', 'float32']. "
                f"Got {self.amp_dtype} instead."
            )

        if self.quantization_strategy is not None:
            assert self.quantization_strategy in ["bnb", "gptq"], (
                "`quantization_strategy` must be one of ['bnb', 'gptq']. "
                f"Got {self.quantization_strategy} instead."
            )
            if self.quantization_strategy == "gptq":
                bits = self.quantization_config.get("bits", None)
                assert bits is not None, (
                    "`quantization_config.bits` must be provided "
                    "when using 'gptq' quantization strategy."
                )
        else:
            self.quantization_config = None

        if self.use_ddp:
            self.ddp_config = OmegaConf.merge(
                self.ddp_config if self.ddp_config is not None else {},
                DEFAULT_DDP_CONFIG,
            )

            # TODO: support multi-node training.
            assert self.ddp_config.max_nodes == 1, (
                "Currently, PyTorch DDP training benchmark "
                "only supports training on a single node."
            )

            assert (
                CUDA_VISIBLE_DEVICES is not None
            ), "Pytorch DDP training benchmark requires CUDA_VISIBLE_DEVICES to be set."
        else:
            self.ddp_config = None


class PyTorchBackend(Backend):
    name: str = "pytorch"
    config: PyTorchConfig

    def __init__(self, model: str, task: str, device: str, hub_kwargs: DictConfig):
        super().__init__(model, task, device, hub_kwargs)
        self.device = torch.device(device)

        LOGGER.info(
            f"\t+ Infered AutoModel class {self.automodel_class.__name__} "
            f"for task {self.task} and model_type {self.model_type}"
        )

    def configure(self, config: PyTorchConfig) -> None:
        super().configure(config)

        # environment options
        if self.config.inter_op_num_threads is not None:
            LOGGER.info(
                "\t+ Setting pytorch "
                f"inter_op_num_threads({self.config.inter_op_num_threads}))"
            )
            torch.set_num_threads(self.config.inter_op_num_threads)
        if self.config.intra_op_num_threads is not None:
            LOGGER.info(
                "\t+ Setting pytorch "
                f"intra_op_num_threads({self.config.intra_op_num_threads}))"
            )
            torch.set_num_interop_threads(self.config.intra_op_num_threads)

        # Load config
        if self.config.torch_dtype is not None:
            if hasattr(torch, self.config.torch_dtype):
                self.config.torch_dtype = getattr(torch, self.config.torch_dtype)

        # Inference config
        if self.config.disable_grad:
            LOGGER.info("\t+ Disabling gradients")
            # everything that comes after this will have its gradients disabled
            torch.set_grad_enabled(False)
        if self.config.amp_dtype is not None:
            if hasattr(torch, self.config.amp_dtype):
                self.config.amp_dtype = getattr(torch, self.config.amp_dtype)

        # Quantization config
        if self.config.quantization_strategy is not None:
            if self.config.quantization_strategy == "gptq":
                self.config.quantization_config = GPTQConfig(
                    **self.config.quantization_config
                )
            elif self.config.quantization_strategy == "bnb":
                self.config.quantization_config = BitsAndBytesConfig(
                    **self.config.quantization_config
                )

        # Load model
        if self.config.no_weights:
            self.load_model_from_config()
        else:
            self.load_model_from_pretrained()

        # Turn on eval mode
        if not self.is_diffusion_pipeline() and self.config.eval_mode:
            LOGGER.info("\t+ Turning on eval mode")
            self.pretrained_model.eval()

        # Turn on BetterTransformer optimizations
        if self.config.bettertransformer:
            LOGGER.info("\t+ Using optimum.bettertransformer")
            self.pretrained_model = BetterTransformer.transform(
                self.pretrained_model,
                keep_original_model=False,
            )

        # Compile model
        if self.config.torch_compile:
            if self.is_diffusion_pipeline():
                LOGGER.info()
                self.pretrained_model.unet = torch.compile(
                    self.pretrained_model.unet,
                    **self.config.torch_compile_kwargs,
                )
            else:
                LOGGER.info("\t+ Using torch.compile on forward pass")
                self.pretrained_model.forward = torch.compile(
                    self.pretrained_model.forward,
                    **self.config.torch_compile_kwargs,
                )

        # DDP config
        if self.config.use_ddp:
            self.config.ddp_config = LaunchConfig(**self.config.ddp_config)

    def load_model_from_pretrained(self) -> None:
        LOGGER.info(f"\t+ Loading pretrained model weights on device: {self.device}")
        if self.is_diffusion_pipeline():
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map,
                **self.hub_kwargs,
            )
            if self.config.device_map is None:
                # Diffusers does not support device_map being a torch.device,
                # thus if not provided we move to device here.
                self.pretrained_model.to(self.device)
        else:
            if self.config.device_map is not None:
                self.pretrained_model = self.automodel_class.from_pretrained(
                    pretrained_model_name_or_path=self.model,
                    quantization_config=self.config.quantization_config,
                    torch_dtype=self.config.torch_dtype,
                    device_map=self.config.device_map,
                    **self.hub_kwargs,
                )
            else:
                with self.device:
                    self.pretrained_model = self.automodel_class.from_pretrained(
                        pretrained_model_name_or_path=self.model,
                        quantization_config=self.config.quantization_config,
                        torch_dtype=self.config.torch_dtype,
                        **self.hub_kwargs,
                    )

    def load_model_from_config(self) -> None:
        # TODO: create no_weights tests

        LOGGER.info("\t+ Initializing empty weights model on device: meta")
        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_config(
                config=self.pretrained_config,
                torch_dtype=self.config.torch_dtype,
                trust_remote_code=self.hub_kwargs.get("trust_remote_code", False),
            )

        if self.config.quantization_strategy is None:
            LOGGER.info(f"\t+ Materializing model on device: {self.device}")
            self.pretrained_model.to_empty(device=self.device)

            LOGGER.info("\t+ Randomizing model weights")
            randomize_weights(self.pretrained_model)
            self.pretrained_model.tie_weights()
        else:
            LOGGER.info("\t+ Materializing model on device: cpu")
            self.pretrained_model.to_empty(device="cpu")

            LOGGER.info("\t+ Randomizing model weights while on device: cpu")
            randomize_weights(self.pretrained_model)
            self.pretrained_model.tie_weights()

            if self.config.quantization_strategy == "bnb":
                quantization_config = BitsAndBytesConfig(**self.quantization_config)
            elif self.config.quantization_strategy == "gptq":
                raise NotImplementedError(
                    "GPTQ requires a pretrained model to be loaded. "
                    "`no_weights` option is not supported with GPTQ."
                )

            from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

            # translating transformers bnb config to accelerate bnb config
            bnb_quantization_config = BnbQuantizationConfig(
                load_in_4bit=quantization_config.load_in_4bit,
                load_in_8bit=quantization_config.load_in_8bit,
                # with dummy_weights, we set this to 0 for reproducibility
                llm_int8_threshold=0,
                torch_dtype=self.config.torch_dtype,
                keep_in_fp32_modules=self.pretrained_model.keep_in_fp32_modules
                if hasattr(self.pretrained_model, "keep_in_fp32_modules")
                else None,
            )

            LOGGER.info("\t+ Quantizing model while on cpu and dispatching to device")
            self.pretrained_model = load_and_quantize_model(
                model=self.pretrained_model,
                bnb_quantization_config=bnb_quantization_config,
                device_map=self.config.device_map
                if self.config.device_map is not None
                else self.device,
            )

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing model for profiling")
        LOGGER.info("\t+ Symbolicly tracing model")
        self.pretrained_model = symbolic_trace(
            model=self.pretrained_model,
            input_names=input_names,
        )

        LOGGER.info("\t+ Wrapping model with FXProfilingWrapper")
        self.pretrained_model = FXProfilingWrapper(self.pretrained_model)

    def forward(self, input: Dict[str, Tensor], **kwargs) -> "ModelOutput":
        with torch.autocast(
            enabled=self.config.amp_autocast,
            device_type=self.device.type,
            dtype=self.config.amp_dtype,
        ):
            output = self.pretrained_model(**input, **kwargs)

        return output

    def generate(self, input: Dict[str, Tensor], **kwargs) -> "ModelOutput":
        with torch.autocast(
            enabled=self.config.amp_autocast,
            device_type=self.device.type,
            dtype=self.config.amp_dtype,
        ):
            output = self.pretrained_model.generate(**input, **kwargs)

        return output

    @record
    def train(
        self,
        training_dataset: "Dataset",
        training_arguments: Dict[str, Any],
        training_callbacks: List["TrainerCallback"],
        training_data_collator: Callable,
    ) -> "TrainerState":
        args = (
            self.config.use_ddp,
            self.pretrained_model,
            training_dataset,
            training_arguments,
            training_callbacks,
            training_data_collator,
        )

        if self.config.use_ddp:
            # For DDP, we log only the stats from the first rank as transformers does.
            # It could make sense to log for all ranks.
            results = elastic_launch(
                config=self.config.ddp_config,
                entrypoint=training_worker,
            )(args)[0]
        else:
            # For DP, we can still use training_worker,
            # simply not wrapped by the elastic_launch class.
            results = training_worker(args)

        return results

    def clean(self) -> None:
        super().clean()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


def training_worker(args) -> "TrainerState":
    use_ddp = args[0]
    pretrained_model = args[1]
    training_dataset = args[2]
    training_arguments = args[3]
    training_callbacks = args[4]
    training_data_collator = args[5]

    if use_ddp:
        LOGGER_WORKER = get_worker_logger("pytorch-ddp-worker", log_all=False)

        env_variables = [
            "RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_MAX_RESTARTS",
        ]

        LOGGER_WORKER.info("Initializing DDP worker")
        for env_var in env_variables:
            LOGGER_WORKER.info(f"{env_var}: {os.environ.get(env_var)}")
    else:
        LOGGER_WORKER = LOGGER

    LOGGER_WORKER.info("\t+ Setting dataset format to `torch`.")
    training_dataset.set_format(
        type="torch", columns=list(training_dataset.features.keys())
    )

    LOGGER_WORKER.info(
        "\t+ Wrapping training arguments with transformers.TrainingArguments"
    )
    training_arguments = TrainingArguments(**training_arguments)

    LOGGER_WORKER.info("\t+ Wrapping model with transformers.Trainer")
    trainer = Trainer(
        model=pretrained_model,
        args=training_arguments,
        callbacks=training_callbacks,
        train_dataset=training_dataset,
        data_collator=training_data_collator,
    )

    LOGGER_WORKER.info("\t+ Starting training")
    trainer.train()
    LOGGER_WORKER.info("\t+ Training finished successfully")
    trainer_state = trainer.state

    return trainer_state
