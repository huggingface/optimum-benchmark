import gc
import os
from collections import OrderedDict
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

import torch
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
    from deepspeed import init_inference

if is_torch_distributed_available():
    import torch.distributed

if is_zentorch_available():
    import zentorch  # type: ignore # noqa: F401


# bachend logger
LOGGER = getLogger("pytorch")


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME = "pytorch"

    def __init__(self, config: PyTorchConfig):
        super().__init__(config)
        self.validate_library()

        # Thread settings
        if self.config.inter_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting pytorch inter_op_num_threads({self.config.inter_op_num_threads}))")
            torch.set_num_threads(self.config.inter_op_num_threads)
        if self.config.intra_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting pytorch intra_op_num_threads({self.config.intra_op_num_threads}))")
            torch.set_num_interop_threads(self.config.intra_op_num_threads)

        # Mixed precision
        if self.config.amp_dtype:
            LOGGER.info(f"\t+ Setting mixed precision dtype to {self.config.amp_dtype}")
            self.amp_dtype = getattr(torch, self.config.amp_dtype)
        else:
            self.amp_dtype = None

        # Quantization
        if self.is_quantized:
            LOGGER.info("\t+ Processing quantization config")
            self.process_quantization_config()
        else:
            self.quantization_config = None

        if self.config.deepspeed_inference:
            if self.quantization_config is not None:
                raise ValueError("Deepspeed-Inference is not compatible with Transformers quantization")

        LOGGER.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights and (self.config.library == "diffusers" or self.config.library == "timm"):
            raise ValueError("Diffusion pipelines and Timm models don't support no weights")
        elif self.config.no_weights:
            LOGGER.info("\t+ Loading model with random weights")
            self.load_model_with_no_weights()
        else:
            LOGGER.info("\t+ Loading model with pretrained weights")
            self.load_model_from_pretrained()

        if self.config.cache_implementation is not None:
            LOGGER.info(f"\t+ Setting cache implementation to {self.config.cache_implementation}")
            self.pretrained_model.generation_config.cache_implementation = self.config.cache_implementation

        # Eval mode
        if self.config.eval_mode and self.config.library != "diffusers":
            LOGGER.info("\t+ Turning on model's eval mode")
            self.pretrained_model.eval()

        # BetterTransformer
        if self.config.to_bettertransformer:
            LOGGER.info("\t+ Enabling BetterTransformer")
            self.pretrained_model.to_bettertransformer()

        # Torch compile
        if self.config.torch_compile:
            if self.config.device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
                LOGGER.info("\t+ Setting float32_matmul_precision to high")
                torch.set_float32_matmul_precision("high")

            if self.config.library == "diffusers":
                LOGGER.info("\t+ Using torch.compile to compile unet and vae")
                self.pretrained_model.unet = torch.compile(
                    self.pretrained_model.unet, **self.config.torch_compile_config
                )
                self.pretrained_model.vae.decode = torch.compile(
                    self.pretrained_model.vae.decode, **self.config.torch_compile_config
                )
            else:
                LOGGER.info("\t+ Using torch.compile on forward pass")
                self.pretrained_model.forward = torch.compile(
                    self.pretrained_model.forward, **self.config.torch_compile_config
                )

        if self.config.peft_type is not None:
            LOGGER.info("\t+ Applying PEFT")
            self.pretrained_model = apply_peft(self.pretrained_model, self.config.peft_type, self.config.peft_config)

        self.tmpdir.cleanup()

    def validate_library(self) -> None:
        if self.config.library == "timm":
            LOGGER.info(f"\t+ Using Timm method {self.automodel_class.__name__}")
        elif self.config.library == "diffusers":
            LOGGER.info(f"\t+ Using Pipeline class {self.automodel_class.__name__}")
        elif self.config.library == "transformers":
            LOGGER.info(f"\t+ Using AutoModel class {self.automodel_class.__name__}")
        else:
            raise ValueError(f"Library {self.config.library} not supported")

    def load_model_from_pretrained(self) -> None:
        if self.config.library == "timm":
            LOGGER.info("\t+ Loading Timm model")
            self.pretrained_model = self.automodel_class(model_name=self.config.model)
            if self.config.device != "cpu":
                LOGGER.info(f"\t+ Moving model to device: {self.config.device}")
                self.pretrained_model.to(self.config.device)
        elif self.config.library == "diffusers":
            LOGGER.info("\t+ Loading Diffusion pipeline")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.config.model,
                pretrained_model_or_path=self.config.model,
                device_map=self.config.device_map,
                **self.config.hub_kwargs,
                **self.automodel_kwargs,
            )
            if self.config.device_map is None and self.config.device != "cpu":
                LOGGER.info(f"\t+ Moving pipeline to device: {self.config.device}")
                self.pretrained_model.to(self.config.device)
        elif self.config.deepspeed_inference:
            if self.config.no_weights:
                with torch.device("meta"):
                    LOGGER.info("\t+ Loading model on meta device for fast initialization")
                    self.pretrained_model = self.automodel_class.from_pretrained(
                        pretrained_model_name_or_path=self.config.model,
                        **self.config.hub_kwargs,
                        **self.automodel_kwargs,
                    )
                LOGGER.info("\t+ Materializing model on CPU")
                self.pretrained_model.to_empty(device="cpu")
                LOGGER.info("\t+ Tying model weights")
                self.pretrained_model.tie_weights()
            else:
                LOGGER.info("\t+ Loading model on cpu to avoid OOM")
                with torch.device("cpu"):
                    self.pretrained_model = self.automodel_class.from_pretrained(
                        pretrained_model_name_or_path=self.config.model,
                        **self.config.hub_kwargs,
                        **self.automodel_kwargs,
                    )

            torch.distributed.barrier()  # better safe than hanging
            LOGGER.info("\t+ Initializing DeepSpeed Inference Engine")
            self.pretrained_model = init_inference(self.pretrained_model, config=self.config.deepspeed_inference_config)
            torch.distributed.barrier()  # better safe than hanging
        elif self.is_quantized:
            # we can't use device context manager on quantized models
            LOGGER.info("\t+ Loading Quantized model")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.config.model,
                device_map=self.config.device_map or torch.device(self.config.device),
                **self.config.hub_kwargs,
                **self.automodel_kwargs,
            )
        elif self.config.device_map is not None:
            # we can't use device context manager since device_map is specified
            LOGGER.info(f"\t+ Loading model with device map: {self.config.device_map}")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.config.model,
                device_map=self.config.device_map,
                **self.config.hub_kwargs,
                **self.automodel_kwargs,
            )
        else:
            LOGGER.info(f"\t+ Loading model directly on device: {self.config.device}")
            with torch.device(self.config.device):
                self.pretrained_model = self.automodel_class.from_pretrained(
                    pretrained_model_name_or_path=self.config.model, **self.config.hub_kwargs, **self.automodel_kwargs
                )

    def create_no_weights_model(self) -> None:
        if self.pretrained_config is None:
            raise ValueError("Can't create no weights model without a pretrained config")

        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights_model")
        LOGGER.info("\t+ Creating no weights model directory")
        os.makedirs(self.no_weights_model, exist_ok=True)
        LOGGER.info("\t+ Creating no weights model state dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()

        if self.is_exllamav2:
            LOGGER.info("\t+ Adding g_idx to no weights model state dict")
            with torch.device("meta"):
                meta_model = self.automodel_class.from_config(self.pretrained_config)
            for name, module in meta_model.named_modules():
                if hasattr(module, "in_features"):
                    state_dict[name + ".g_idx"] = torch.ones((module.in_features,), dtype=torch.int32)

        LOGGER.info("\t+ Saving no weights model safetensors")
        safetensors = os.path.join(self.no_weights_model, "model.safetensors")
        save_file(tensors=state_dict, filename=safetensors, metadata={"format": "pt"})

        if self.is_quantized:
            LOGGER.info("\t+ Adding quantization config to no weights model's pretrained config")
            self.pretrained_config.quantization_config = self.quantization_config.to_dict()
            # tricking from_pretrained to load the model as if it was quantized

        LOGGER.info("\t+ Saving no weights model pretrained config")
        if self.config.library == "transformers":
            self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)

    def load_model_with_no_weights(self) -> None:
        LOGGER.info("\t+ Creating no weights model")
        self.create_no_weights_model()

        with random_init_weights():
            original_model, self.config.model = self.config.model, self.no_weights_model
            LOGGER.info("\t+ Loading no weights AutoModel")
            self.load_model_from_pretrained()
            self.config.model = original_model

    def process_quantization_config(self) -> None:
        if self.is_gptq_quantized:
            LOGGER.info("\t+ Processing GPTQ config")
            self.quantization_config = GPTQConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        elif self.is_awq_quantized:
            LOGGER.info("\t+ Processing AWQ config")
            self.quantization_config = AwqConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        elif self.is_bnb_quantized:
            LOGGER.info("\t+ Processing BitsAndBytes config")
            self.quantization_config = BitsAndBytesConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        else:
            raise ValueError(f"Quantization scheme {self.config.quantization_scheme} not recognized")

    @property
    def is_quantized(self) -> bool:
        return self.config.quantization_scheme is not None or hasattr(self.pretrained_config, "quantization_config")

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
        return (self.is_gptq_quantized or self.is_awq_quantized) and (
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

    @property
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.is_quantized:
            kwargs["quantization_config"] = self.quantization_config

        if self.config.torch_dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)

        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation

        if self.config.low_cpu_mem_usage is not None:
            kwargs["low_cpu_mem_usage"] = self.config.low_cpu_mem_usage

        if self.config.no_weights:
            # we use our own context manager to load the model with random weights
            kwargs["_fast_init"] = False

        return kwargs

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = super().prepare_inputs(inputs)

        if self.config.library == "diffusers":
            inputs = {"prompt": inputs["prompt"]}
        elif self.config.library == "timm":
            inputs = {"x": inputs["pixel_values"].to(self.config.device)}
        else:
            for key, value in inputs.items():
                inputs[key] = value.to(self.config.device)

        return inputs

    @torch.inference_mode()
    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        with torch.autocast(device_type=self.config.device, dtype=self.amp_dtype, enabled=self.config.amp_autocast):
            return self.pretrained_model.forward(**inputs, **kwargs)

    @torch.inference_mode()
    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        with torch.autocast(device_type=self.config.device, dtype=self.amp_dtype, enabled=self.config.amp_autocast):
            return self.pretrained_model.generate(**inputs, **kwargs)

    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        with torch.autocast(device_type=self.config.device, dtype=self.amp_dtype, enabled=self.config.amp_autocast):
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
        LOGGER.info(f"\t+ Wrapping training arguments with {TrainingArguments.__name__}")
        training_arguments = TrainingArguments(**training_arguments)
        LOGGER.info(f"\t+ Wrapping model with {Trainer.__name__}")
        trainer = Trainer(
            args=training_arguments,
            model=self.pretrained_model,
            callbacks=training_callbacks,
            train_dataset=training_dataset,
            data_collator=training_data_collator,
        )
        LOGGER.info("\t+ Starting training")
        trainer.train()
        LOGGER.info("\t+ Finished training")

    def seed(self):
        super().seed()
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            LOGGER.info("\t+ Cleaning backend temporary directory")
            self.tmpdir.cleanup()

        gc.collect()
