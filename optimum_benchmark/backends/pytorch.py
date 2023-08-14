from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from logging import getLogger
from datasets import Dataset
from torch import Tensor
import torch
import os
import time

from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torch.distributed.elastic.multiprocessing import Std
import logging.config

from transformers.utils import ModelOutput
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.utils.fx import symbolic_trace
from transformers.trainer_utils import TrainOutput
from optimum.bettertransformer import BetterTransformer

from optimum_benchmark.backends.base import Backend, BackendConfig
from optimum_benchmark.profilers.fx_profiler import FXProfilingWrapper

if TYPE_CHECKING:
    from transformers import TrainerState, TrainerControl

WARMUP_STEPS = 40

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
    device_map: Optional[str] = None

    # quantization options
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # optimization options
    bettertransformer: bool = False

    # compilation options
    torch_compile: bool = False
    torch_compile_config: Dict = field(default_factory=lambda: {
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
        if self.task not in ["stable-diffusion", "stable-diffusion-xl"]:
            if config.load_in_8bit or config.load_in_4bit or config.device_map is not None:
                self.pretrained_model = self.automodel_class.from_pretrained(
                    pretrained_model_name_or_path=self.model,
                    torch_dtype=self.torch_dtype,
                    device_map=config.device_map if config.device_map is not None else self.device,
                    load_in_8bit=config.load_in_8bit,
                    load_in_4bit=config.load_in_4bit,
                    llm_int8_threshold=0,
                    **self.hub_kwargs,
                )
            else:
                # When a device_map is not specified, we do not rely on accelerate to load the load and rather try PyTorch-native context.
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
                device_map=config.device_map,
                **self.hub_kwargs,
            )
            if config.device_map is None:
                # Diffusers does not support device_map being a torch.device, thus if not provided, move to device here.
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

    def forward(self, input: Dict[str, Tensor], **kwargs) -> ModelOutput:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_autocast,
        ):
            output = self.pretrained_model(**input, **kwargs)

        return output

    def generate(self, input: Dict[str, Tensor], **kwargs) -> ModelOutput:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_autocast,
        ):
            output = self.pretrained_model.generate(**input, **kwargs)

        return output

    def train(self) -> None:
        raise Exception("For PyTorch backend training, please call backend.run_pytorch_training.")

    def run_pytorch_training(self, training_config, training_arguments, training_dataset, training_data_collator):
        LOGGER.info("Running training benchmark")

        # Converting from DictConfig to Dict is required to avoid a warning with DDP:
        # `[W CudaIPCTypes.cpp:15] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]`
        training_arguments_dict = OmegaConf.to_container(training_arguments, resolve=True)
        
        if training_config.use_ddp:
            # TODO: support multi-node training. Hydra is probably not the good infra for that though.
            if training_config.ddp_config.max_nodes != 1:
                raise ValueError("PyTorch DDP training benchmark currently supports only training on a single node.")
                
            launch_config = LaunchConfig(**training_config.ddp_config)
            LOGGER.info(f"PyTorch DDP launch config: {launch_config}")
            
            # TODO: The backend instance can not be passed here (cannot pickle 'weakref' object) so the nn.Module is passed directly.
            # It is not clear who is using weakref though.
            results = elastic_launch(
                config=launch_config,
                entrypoint=ddp_callable,
            )((self.pretrained_model, training_dataset, training_arguments_dict, training_data_collator, True))
            
            # For DDP, we log only the stats from the first rank as transformers does. It could make sense to log for all ranks.
            results = results[0]
        else:
            # For simple Data Parallel, we can still use ddp_callable, simply not wrapped by the elastic_launch class.
            results = ddp_callable((self.pretrained_model, training_dataset, training_arguments_dict, training_data_collator, False))
        
        return results


def get_logger(name: Optional[str] = None, log_all: bool = False):
    """
    PyTorch DDP subprocesses do not inherit from Hydra logger. Thus, we need to reconfigure the logger for the workers.
    """
    if os.environ["RANK"] == "0" or log_all:
        # TODO: also configure logging for other ranks
        hydra_conf = OmegaConf.load('.hydra/hydra.yaml')
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    return getLogger(name)

# Adapted from transformers.trainer_utils.speed_metrics
def speed_metrics(trainer):
    """
    Measure and return speed performance metrics.
    """
    # Reference: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L1559
    total_train_batch_size = trainer._train_batch_size * trainer.args.gradient_accumulation_steps * trainer.args.world_size
    result = {}

    # Warmup metrics.
    num_warmup_steps = WARMUP_STEPS
    num_warmup_samples = num_warmup_steps * total_train_batch_size
    warmup_runtime = trainer.state.warmup_end - trainer.state.warmup_start

    warmup_samples_per_second = num_warmup_samples / warmup_runtime
    result["warmup_runtime"] = warmup_runtime
    result["warmup_samples_per_second"] = round(warmup_samples_per_second, 3)
    warmup_steps_per_second = num_warmup_steps / warmup_runtime
    result["warmup_steps_per_second"] = round(warmup_steps_per_second, 3)

    # Training metrics.
    num_train_steps = trainer.state.max_steps - WARMUP_STEPS
    num_train_samples = num_train_steps * total_train_batch_size
    train_runtime = trainer.state.training_end - trainer.state.training_start

    train_samples_per_second = num_train_samples / train_runtime
    result["train_runtime"] = train_runtime
    result["train_samples_per_second"] = round(train_samples_per_second, 3)
    train_steps_per_second = num_train_steps / train_runtime
    result["train_steps_per_second"] = round(train_steps_per_second, 3)

    return result

class MeasurementCallback(TrainerCallback):
    def on_step_begin(self, args: TrainingArguments, state: "TrainerState", control: "TrainerControl", **kwargs):
        if state.global_step == 0:
            # This check is here because max_steps is set only once the training is launched, thus we can not check before calling trainer.train().
            if state.max_steps <= WARMUP_STEPS:
                raise ValueError(f"Total training steps {state.max_steps} is smaller than the number of warmup steps {WARMUP_STEPS}. Please increase the total number of steps (for example by increasing the dataset size).")

            state.warmup_start = time.time_ns() * 1e-9
        elif state.global_step == WARMUP_STEPS:
            state.warmup_end = time.time_ns() * 1e-9
            state.training_start = time.time_ns() * 1e-9
        elif state.global_step == state.max_steps - 1:
            state.training_end = time.time_ns() * 1e-9
        elif state.global_step > state.max_steps - 1:
            raise ValueError("global_step > state.max_steps - 1")

def ddp_callable(args):
    pretrained_model = args[0]
    training_dataset = args[1]
    training_arguments = args[2]
    training_data_collator = args[3]
    use_ddp = args[4]

    if use_ddp:
        LOGGER_WORKER = get_logger("training-ddp-worker", log_all=False)

        env_variables = [
            "RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_MAX_RESTARTS",
        ]
        for env_var in env_variables:
            LOGGER_WORKER.info(f"{env_var}: {os.environ.get(env_var)}")
    else:
        LOGGER_WORKER = LOGGER

    LOGGER_WORKER.info("\t+ Wrapping model with transformers.Trainer")
    training_arguments = TrainingArguments(**training_arguments)

    trainer = Trainer(
        model=pretrained_model,
        train_dataset=training_dataset,
        data_collator=training_data_collator,
        args=training_arguments,
        callbacks=[MeasurementCallback]
    )
    
    LOGGER_WORKER.info("Training model")
    trainer.train()
    results = speed_metrics(trainer)

    return results
