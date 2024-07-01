import os
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset
from safetensors.torch import save_file
from transformers import (
    AwqConfig,
    BitsAndBytesConfig,
    GPTQConfig,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainingArguments,
)

from ...import_utils import is_deepspeed_available, is_torch_distributed_available, is_zentorch_available
from ..base import Backend
from ..peft_utils import apply_peft
from ..transformers_utils import random_init_weights
from .config import PyTorchConfig

if is_deepspeed_available():
    import deepspeed

if is_torch_distributed_available():
    import torch.distributed

if is_zentorch_available():
    import zentorch  # type: ignore # noqa: F401


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME = "pytorch"

    def __init__(self, config: PyTorchConfig):
        super().__init__(config)
        self.validate_library()

        if self.config.deepspeed_inference and self.is_quantized:
            raise ValueError("Deepspeed-Inference is not compatible with Transformers quantization")

        # Quantization
        if self.is_quantized:
            self.logger.info("\t+ Processing quantization config")
            self.process_quantization_config()

        # Threads
        if self.config.inter_op_num_threads is not None:
            self.logger.info(f"\t+ Setting pytorch inter_op_num_threads({self.config.inter_op_num_threads}))")
            torch.set_num_threads(self.config.inter_op_num_threads)
        if self.config.intra_op_num_threads is not None:
            self.logger.info(f"\t+ Setting pytorch intra_op_num_threads({self.config.intra_op_num_threads}))")
            torch.set_num_interop_threads(self.config.intra_op_num_threads)

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

        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        # Model
        if self.config.no_weights and (self.config.library == "diffusers" or self.config.library == "timm"):
            raise ValueError("Diffusion pipelines and Timm models don't support no weights")
        elif self.config.no_weights:
            self.logger.info("\t+ Loading model with random weights")
            self.load_model_with_no_weights()
        else:
            self.logger.info("\t+ Loading model with pretrained weights")
            self.load_model_from_pretrained()

        self.tmpdir.cleanup()

        # KV-Cache
        if self.config.cache_implementation is not None:
            self.logger.info(f"\t+ Setting cache implementation to {self.config.cache_implementation}")
            self.pretrained_model.generation_config.cache_implementation = self.config.cache_implementation

        # Eval mode
        if self.config.eval_mode and self.config.library != "diffusers":
            self.logger.info("\t+ Turning on model's eval mode")
            self.pretrained_model.eval()

        # BetterTransformer
        if self.config.to_bettertransformer:
            self.logger.info("\t+ Enabling BetterTransformer")
            self.pretrained_model.to_bettertransformer()

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
            if self.config.library == "diffusers":
                self.logger.info("\t+ Using torch.compile on unet and vae")
                self.pretrained_model.unet = torch.compile(
                    self.pretrained_model.unet, **self.config.torch_compile_config
                )
                self.pretrained_model.vae.decode = torch.compile(
                    self.pretrained_model.vae.decode, **self.config.torch_compile_config
                )
            else:
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

    def validate_library(self) -> None:
        if self.config.library == "timm":
            self.logger.info(f"\t+ Using Timm's {self.automodel_class.__name__}")
        elif self.config.library == "diffusers":
            self.logger.info(f"\t+ Using Diffusers Pipeline {self.automodel_class.__name__}")
        elif self.config.library == "transformers":
            self.logger.info(f"\t+ Using AutoModel {self.automodel_class.__name__}")
        else:
            raise ValueError(f"Library {self.config.library} not supported")

    def load_model_from_pretrained(self) -> None:
        if self.config.library == "timm":
            self.logger.info("\t+ Loading Timm model")
            self.pretrained_model = self.automodel_class(model_name=self.config.model)
            if self.config.device != "cpu":
                self.logger.info(f"\t+ Moving Timm model to device: {self.config.device}")
                self.pretrained_model = self.pretrained_model.to(self.config.device)

        elif self.config.library == "diffusers":
            self.logger.info("\t+ Loading Diffusion Pipeline")
            self.pretrained_model = self.automodel_class.from_pretrained(
                # pretrained_model_name_or_path=self.config.model,
                # pretrained_model_or_path=self.config.model,
                self.config.model,
                device_map=self.config.device_map,
                **self.config.model_kwargs,
                **self.automodel_kwargs,
            )
            if self.config.device_map is None and self.config.device != "cpu":
                self.logger.info(f"\t+ Moving Diffusion Pipeline to device: {self.config.device}")
                self.pretrained_model = self.pretrained_model.to(self.config.device)

        elif self.is_quantized:
            self.logger.info(f"\t+ Loading {self.quantization_config.quant_method}-quantized model")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.config.model,
                device_map=self.config.device_map or torch.device(self.config.device),
                # quantized models are more compatible with device_map dispatcher than (to(device))
                # using to(device) on quantized models sometimes leaves some layers on cpu or raises
                # an error because the layers are already on the device
                **self.config.model_kwargs,
                **self.automodel_kwargs,
            )

        elif self.config.device_map is not None:
            self.logger.info(f"\t+ Loading Transformers model with device map: {self.config.device_map}")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.config.model,
                device_map=self.config.device_map,
                **self.config.model_kwargs,
                **self.automodel_kwargs,
            )

        else:
            self.logger.info("\t+ Loading Transformers model")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.config.model, **self.config.model_kwargs, **self.automodel_kwargs
            )
            if self.config.device != "cpu":
                self.logger.info(f"\t+ Moving Transformers model to device: {self.config.device}")
                self.pretrained_model = self.pretrained_model.to(self.config.device)

    def create_no_weights_model(self) -> None:
        if self.pretrained_config is None:
            raise ValueError("Can't create no weights model without a pretrained config")

        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights_model")
        self.logger.info("\t+ Creating no weights model directory")
        os.makedirs(self.no_weights_model, exist_ok=True)
        self.logger.info("\t+ Creating no weights model state dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()

        if self.is_exllamav2:
            self.logger.info("\t+ Adding g_idx to no weights model state dict")
            with torch.device("meta"):
                meta_model = self.automodel_class.from_config(self.pretrained_config)
            for name, module in meta_model.named_modules():
                if hasattr(module, "in_features"):
                    state_dict[name + ".g_idx"] = torch.ones((module.in_features,), dtype=torch.int32)

        self.logger.info("\t+ Saving no weights model safetensors")
        safetensors = os.path.join(self.no_weights_model, "model.safetensors")
        save_file(tensors=state_dict, filename=safetensors, metadata={"format": "pt"})

        if self.is_quantized:
            self.logger.info("\t+ Adding quantization config to no weights model's pretrained config")
            self.pretrained_config.quantization_config = self.quantization_config.to_dict()
            # tricking from_pretrained to load the model as if it was quantized

        self.logger.info("\t+ Saving no weights model pretrained config")
        self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)

    def load_model_with_no_weights(self) -> None:
        self.create_no_weights_model()
        original_model, self.config.model = self.config.model, self.no_weights_model

        if self.config.deepspeed_inference:
            with torch.device("meta"):
                # with big models, loading no_weights_model is very slow (randomizing every weight)
                # so we load the model on meta device to speed up the process and then move it to cpu
                self.logger.info("\t+ Loading Transformers model on meta device for fast initialization")
                self.pretrained_model = self.automodel_class.from_pretrained(
                    pretrained_model_name_or_path=self.config.model,
                    **self.config.model_kwargs,
                    **self.automodel_kwargs,
                )
            self.logger.info("\t+ Materializing meta model on CPU to avoid OOM")
            self.pretrained_model.to_empty(device="cpu")

        elif not self.is_quantized and self.config.device_map is None:
            with torch.device(self.config.device):
                self.logger.info("\t+ Loading Transformers model using device context manager for fast initialization")
                self.pretrained_model = self.automodel_class.from_pretrained(
                    pretrained_model_name_or_path=self.no_weights_model,
                    **self.config.model_kwargs,
                    **self.automodel_kwargs,
                )

        else:
            with random_init_weights():
                self.load_model_from_pretrained()

        self.config.model = original_model

    def process_quantization_config(self) -> None:
        if self.is_gptq_quantized:
            self.logger.info("\t+ Processing GPTQ config")
            self.quantization_config = GPTQConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        elif self.is_awq_quantized:
            self.logger.info("\t+ Processing AWQ config")
            self.quantization_config = AwqConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        elif self.is_bnb_quantized:
            self.logger.info("\t+ Processing BitsAndBytes config")
            self.quantization_config = BitsAndBytesConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        else:
            raise ValueError(f"Quantization scheme {self.config.quantization_scheme} not recognized")

    @property
    def is_distributed(self) -> bool:
        return is_torch_distributed_available() and torch.distributed.is_initialized()

    @property
    def is_tp_distributed(self) -> bool:
        return self.is_distributed and self.config.deepspeed_inference

    @property
    def is_dp_distributed(self) -> bool:
        return self.is_distributed and not self.config.deepspeed_inference

    @property
    def is_quantized(self) -> bool:
        return self.config.quantization_scheme is not None or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method", None) is not None
        )

    @property
    def is_bnb_quantized(self) -> bool:
        return self.config.quantization_scheme == "bnb" or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method", None) == "bnb"
        )

    @property
    def is_gptq_quantized(self) -> bool:
        return self.config.quantization_scheme == "gptq" or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method", None) == "gptq"
        )

    @property
    def is_awq_quantized(self) -> bool:
        return self.config.quantization_scheme == "awq" or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method", None) == "awq"
        )

    @property
    def is_exllamav2(self) -> bool:
        return (
            self.is_quantized
            and (self.is_gptq_quantized or self.is_awq_quantized)
            and (
                (
                    hasattr(self.pretrained_config, "quantization_config")
                    and hasattr(self.pretrained_config.quantization_config, "exllama_config")
                    and self.pretrained_config.quantization_config.exllama_config.get("version", None) == 2
                )
                or (
                    "exllama_config" in self.config.quantization_config
                    and self.config.quantization_config["exllama_config"].get("version", None) == 2
                )
            )
        )

    @property
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.torch_dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)

        if self.is_quantized:
            kwargs["quantization_config"] = self.quantization_config

        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation

        if self.config.low_cpu_mem_usage is not None:
            kwargs["low_cpu_mem_usage"] = self.config.low_cpu_mem_usage

        if self.config.no_weights:
            # we use our own context manager to load the model with random weights
            kwargs["_fast_init"] = False

        return kwargs

    def prepare_inputs(
        self, inputs: Dict[str, Any], input_shapes: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, input_shapes = super().prepare_inputs(inputs, input_shapes)

        if self.is_dp_distributed:
            if input_shapes["batch_size"] % torch.distributed.get_world_size() != 0:
                raise ValueError(
                    f"Batch size {input_shapes['batch_size']} must be divisible by data parallel "
                    f"world size {torch.distributed.get_world_size()}"
                )
            with Accelerator().split_between_processes(inputs=inputs, apply_padding=False) as split_inputs:
                input_shapes["batch_size"] = input_shapes["batch_size"] // torch.distributed.get_world_size()
                inputs = split_inputs

        if self.is_tp_distributed:
            if torch.distributed.get_rank() != 0:
                # this is to force throughput of non main shards to 0
                input_shapes["batch_size"] = 0

        if self.config.library == "timm":
            inputs = {"x": inputs["pixel_values"]}

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.config.device)

        return inputs, input_shapes

    @torch.inference_mode()
    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.forward(**inputs, **kwargs)

    @torch.inference_mode()
    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
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
