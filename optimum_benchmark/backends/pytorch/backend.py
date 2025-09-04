from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

import torch
from accelerate import Accelerator

# from accelerate.utils import compile_regions
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainerState, TrainingArguments
from transformers.quantizers import AutoQuantizationConfig

from ...import_utils import (
    is_accelerate_available,
    is_deepspeed_available,
    is_gptqmodel_available,
    is_torch_distributed_available,
    is_zentorch_available,
)
from ..base import Backend
from ..peft_utils import apply_peft
from ..transformers_utils import fast_weights_init
from .config import PyTorchConfig

if is_accelerate_available():
    from accelerate.utils import compile_regions

if is_deepspeed_available():
    import deepspeed  # type: ignore

if is_torch_distributed_available():
    import torch.distributed  # type: ignore

if is_zentorch_available():
    import zentorch  # type: ignore # noqa: F401

if is_gptqmodel_available():
    import enum

    if not hasattr(enum, "EnumType") and hasattr(enum, "EnumMeta"):
        # This is a workaround for a bug in gptqmodel where it tries to access EnumType
        # from the enum module, but it is not available in Python 3.10 and below.
        enum.EnumType = enum.EnumMeta


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME = "pytorch"

    def __init__(self, config: PyTorchConfig):
        super().__init__(config)

        # Threads
        if self.config.inter_op_num_threads is not None:
            self.logger.info(f"\t+ Setting pytorch inter_op_num_threads({self.config.inter_op_num_threads}))")
            torch.set_num_threads(self.config.inter_op_num_threads)

        if self.config.intra_op_num_threads is not None:
            self.logger.info(f"\t+ Setting pytorch intra_op_num_threads({self.config.intra_op_num_threads}))")
            torch.set_num_interop_threads(self.config.intra_op_num_threads)

        # TF32
        if self.config.allow_tf32:
            self.logger.info("\t+ Enabling TF32")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Autocast
        if self.config.autocast_enabled:
            self.logger.info("\t+ Enabling automatic mixed precision")
            torch.set_autocast_enabled(True)

            if self.config.autocast_dtype is not None:
                if self.config.device == "cpu":
                    self.logger.info(f"\t+ Setting autocast cpu dtype to {self.config.autocast_dtype}")
                    torch.set_autocast_cpu_dtype(getattr(torch, self.config.autocast_dtype))
                elif self.config.device == "cuda":
                    self.logger.info(f"\t+ Setting autocast gpu dtype to {self.config.autocast_dtype}")
                    torch.set_autocast_gpu_dtype(getattr(torch, self.config.autocast_dtype))
                else:
                    raise ValueError(f"Device {self.config.device} not supported for autocast")

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.library == "transformers":
            self.load_transformers_model()
        elif self.config.library == "diffusers":
            self.load_diffusers_model()
        elif self.config.library == "timm":
            self.load_timm_model()
        else:
            raise ValueError(f"Library {self.config.library} not supported for PyTorch backend")

        self.logger.info("\t+ Cleaning up backend temporary directory")
        self.tmpdir.cleanup()

    def load_transformers_model_from_pretrained(self) -> None:
        self.logger.info("\t+ Loading Transformers model")
        self.pretrained_model = self.automodel_loader.from_pretrained(
            pretrained_model_name_or_path=self.config.model,
            **self.config.model_kwargs,
            **self.automodel_kwargs,
        )
        if self.config.device_map is None and self.config.device != "cpu":
            self.logger.info(f"\t+ Moving Transformers model to device: {self.config.device}")
            self.pretrained_model = self.pretrained_model.to(self.config.device)

    def load_transformers_model_with_no_weights(self) -> None:
        with fast_weights_init():
            original_model, self.config.model = self.config.model, self.no_weights_model_path.as_posix()
            self.load_transformers_model_from_pretrained()
            self.config.model = original_model

    def load_transformers_model(self):
        if self.config.deepspeed_inference and self.is_quantized:
            raise ValueError("Deepspeed-Inference is not compatible with Transformers quantization")

        # Quantization
        if self.is_quantized:
            self.logger.info("\t+ Processing AutoQuantization config")
            self.quantization_config = AutoQuantizationConfig.from_dict(
                dict(
                    getattr(self.pretrained_config, "quantization_config", {}),
                    **self.config.quantization_config,
                )
            )

        # Model loading
        if self.config.no_weights:
            self.logger.info("\t+ Creating no weights model")
            if self.config.tp_plan is not None:
                self.create_no_weights_model_slow()
            else:
                self.create_no_weights_model_fast()
            self.logger.info("\t+ Loading model with random weights")
            self.load_transformers_model_with_no_weights()
        else:
            self.logger.info("\t+ Loading model with pretrained weights")
            self.load_transformers_model_from_pretrained()

        # KV-Cache
        if self.config.cache_implementation is not None:
            self.logger.info(f"\t+ Setting cache implementation to {self.config.cache_implementation}")
            self.pretrained_model.generation_config.cache_implementation = self.config.cache_implementation

        # BetterTransformer
        if self.config.to_bettertransformer:
            self.logger.info("\t+ To BetterTransformer")
            self.pretrained_model.to_bettertransformer()

        # Eval mode
        if self.config.eval_mode:
            self.logger.info("\t+ Enabling eval mode")
            self.pretrained_model.eval()

        # PEFT
        if self.config.peft_type is not None:
            self.logger.info("\t+ Applying PEFT")
            self.pretrained_model = apply_peft(self.pretrained_model, self.config.peft_type, self.config.peft_config)

        # DeepSpeed
        if self.config.deepspeed_inference:
            self.logger.info("\t+ Initializing DeepSpeed Inference Engine")
            self.pretrained_model = deepspeed.init_inference(
                model=self.pretrained_model, config=self.config.deepspeed_inference_config
            )

        # Torch compile
        if self.config.torch_compile:
            if self.config.torch_compile_target == "model":
                self.logger.info("\t+ Using torch.compile on model")
                self.pretrained_model = torch.compile(self.pretrained_model, **self.config.torch_compile_config)
            elif self.config.torch_compile_target == "regions":
                self.logger.info("\t+ Using accelerate.utils.compile_regions on model")
                self.pretrained_model = compile_regions(self.pretrained_model, **self.config.torch_compile_config)
            elif self.config.torch_compile_target == "forward":
                self.logger.info("\t+ Using torch.compile on forward")
                self.pretrained_model.forward = torch.compile(
                    self.pretrained_model.forward, **self.config.torch_compile_config
                )
            else:
                raise ValueError(f"Target {self.config.torch_compile_target} not supported")

    def load_diffusers_pipeline_from_pretrained(self) -> None:
        self.pretrained_model = self.automodel_loader.from_pretrained(
            self.config.model,
            # pretrained_model_name_or_path=self.config.model,
            # pretrained_model_or_path=self.config.model,
            device_map=self.config.device_map,
            **self.config.model_kwargs,
            **self.automodel_kwargs,
        )
        if self.config.device_map is None and self.config.device != "cpu":
            self.logger.info(f"\t+ Moving Diffusion Pipeline to device: {self.config.device}")
            self.pretrained_model = self.pretrained_model.to(self.config.device)

    def load_diffusers_model(self):
        self.logger.info("\t+ Loading Diffusion Pipeline")
        self.logger.info(f"\t+ Using Diffusers Pipeline {self.automodel_loader.__name__}")

        # Model loading
        if self.config.no_weights:
            raise ValueError("No weights model not supported for Diffusers")
        else:
            self.load_diffusers_pipeline_from_pretrained()

        # Torch compile
        if self.config.torch_compile:
            self.logger.info("\t+ Using torch.compile on unet and vae")
            self.pretrained_model.unet = torch.compile(self.pretrained_model.unet, **self.config.torch_compile_config)
            self.pretrained_model.vae.decode = torch.compile(
                self.pretrained_model.vae.decode, **self.config.torch_compile_config
            )

    def load_timm_model_form_pretrained(self) -> None:
        self.pretrained_model = self.automodel_loader(model_name=self.config.model)
        if self.config.device != "cpu":
            self.logger.info(f"\t+ Moving Timm model to device: {self.config.device}")
            self.pretrained_model = self.pretrained_model.to(self.config.device)

    def load_timm_model(self):
        self.logger.info("\t+ Loading Timm model")
        self.logger.info(f"\t+ Using Timm's {self.automodel_loader.__name__}")

        # Model loading
        if self.config.no_weights:
            raise ValueError("No weights model not supported for Timm")
        else:
            self.load_timm_model_form_pretrained()

        # Torch compile
        if self.config.torch_compile:
            if self.config.torch_compile_target == "forward":
                self.logger.info("\t+ Using torch.compile on forward")
                self.pretrained_model.forward = torch.compile(
                    self.pretrained_model.forward, **self.config.torch_compile_config
                )
            elif self.config.torch_compile_target == "model":
                self.logger.info("\t+ Using torch.compile on model")
                self.pretrained_model = torch.compile(self.pretrained_model, **self.config.torch_compile_config)
            else:
                raise ValueError(f"Target {self.config.torch_compile_target} not supported")

    @property
    def is_quantized(self) -> bool:
        return self.config.quantization_scheme is not None or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method") is not None
        )

    @property
    def is_gptq_quantized(self) -> bool:
        return self.config.quantization_scheme == "gptq" or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method") == "gptq"
        )

    @property
    def is_bnb_quantized(self) -> bool:
        return self.config.quantization_scheme == "bnb" or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method") == "bnb"
        )

    @property
    def is_exllamav2(self) -> bool:
        return (
            self.is_quantized
            and (self.is_gptq_quantized)
            and (
                (
                    hasattr(self.pretrained_config, "quantization_config")
                    and hasattr(self.pretrained_config.quantization_config, "exllama_config")
                    and self.pretrained_config.quantization_config.exllama_config.get("version") == 2
                )
                or (
                    "exllama_config" in self.config.quantization_config
                    and self.config.quantization_config["exllama_config"].get("version") == 2
                )
            )
        )

    @property
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.torch_dtype is not None:
            if hasattr(torch, self.config.torch_dtype):
                kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)
            else:
                kwargs["torch_dtype"] = self.config.torch_dtype

        if self.is_quantized:
            kwargs["quantization_config"] = self.quantization_config

        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation

        if self.config.low_cpu_mem_usage is not None:
            kwargs["low_cpu_mem_usage"] = self.config.low_cpu_mem_usage

        if self.config.device_map is not None:
            kwargs["device_map"] = self.config.device_map

        if self.config.tp_plan is not None:
            kwargs["tp_plan"] = self.config.tp_plan

        return kwargs

    @property
    def split_between_processes(self) -> bool:
        return (
            is_torch_distributed_available()
            and torch.distributed.is_initialized()
            # we don't split between processes if tp (native or deepspeed) is used
            and not self.config.deepspeed_inference
            and self.config.tp_plan is None
        )

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.split_between_processes:
            with Accelerator().split_between_processes(inputs=inputs, apply_padding=False) as process_inputs:
                inputs = process_inputs

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.config.device)

        if self.config.library == "timm":
            inputs = {"x": inputs["pixel_values"]}

        return inputs

    @torch.inference_mode()
    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.forward(**inputs, **kwargs)

    @torch.inference_mode()
    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        assert kwargs.get("max_new_tokens") == kwargs.get("min_new_tokens") == 1, (
            "For prefilling, max_new_tokens and min_new_tokens must be equal to 1"
        )
        return self.pretrained_model.generate(**inputs, **kwargs)

    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(**inputs, **kwargs)

    @torch.inference_mode()
    def call(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model(**inputs, **kwargs)

    def train(
        self,
        training_dataset: Dataset,
        training_arguments: Dict[str, Any],
        training_callbacks: List[TrainerCallback],
        training_data_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
    ) -> TrainerState:
        self.logger.info(f"\t+ Wrapping training arguments with {TrainingArguments.__name__}")
        training_arguments["use_cpu"] = self.config.device == "cpu"
        training_arguments = TrainingArguments(**training_arguments)
        self.logger.info(f"\t+ Wrapping model with {Trainer.__name__}")
        trainer = Trainer(
            args=training_arguments,
            model=self.pretrained_model,
            callbacks=training_callbacks,
            train_dataset=training_dataset,
            data_collator=training_data_collator,
        )
        self.logger.info("\t+ Starting training")
        trainer.train()
        self.logger.info("\t+ Finished training")
