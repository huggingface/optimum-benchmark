import gc
import os
from logging import getLogger
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

from ..base import Backend
from .config import PyTorchConfig
from ..peft_utils import get_peft_config_class
from ..transformers_utils import randomize_weights
from ...import_utils import is_deepspeed_available, is_peft_available

import torch
from datasets import Dataset
from safetensors.torch import save_file
import datasets.utils.logging as datasets_logging
from transformers.modeling_utils import no_init_weights
import transformers.utils.logging as transformers_logging
from transformers import TrainerCallback, TrainerState, Trainer, TrainingArguments

if is_peft_available():
    from peft import get_peft_model  # type: ignore

if is_deepspeed_available():
    from deepspeed import init_inference  # type: ignore

# disable other loggers
datasets_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

# bachend logger
LOGGER = getLogger("pytorch")


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME = "pytorch"

    def __init__(self, config: PyTorchConfig):
        super().__init__(config)
        self.validate_library()

        # Threads
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

        LOGGER.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights and self.config.library == "diffusers":
            raise ValueError("Diffusion pipelines are not supported with no_weights=True")
        elif self.config.no_weights:
            LOGGER.info("\t+ Loading model with no weights")
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
            if self.config.library == "diffusers":
                LOGGER.info("\t+ Using torch.compile on unet forward pass")
                # TODO: should we compile vae and/or clip as well ?
                self.pretrained_model.unet.forward = torch.compile(
                    self.pretrained_model.unet.forward, **self.config.torch_compile_config
                )
            else:
                LOGGER.info("\t+ Using torch.compile on forward pass")
                self.pretrained_model.forward = torch.compile(
                    self.pretrained_model.forward, **self.config.torch_compile_config
                )

        if self.config.peft_strategy is not None:
            LOGGER.info("\t+ Using PEFT")
            peft_config_class = get_peft_config_class(self.config.peft_strategy)
            peft_config = peft_config_class(**self.config.peft_config)
            self.pretrained_model = get_peft_model(self.pretrained_model, peft_config=peft_config)

        if self.config.deepspeed_inference:
            LOGGER.info("\t+ Using DeepSpeed-Inference")
            self.pretrained_model = init_inference(
                self.pretrained_model,
                config=self.config.deepspeed_inference_config,
                dtype=getattr(self.pretrained_model, "dtype", None),
            )

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
            self.pretrained_model = self.automodel_class(self.config.model)
            self.pretrained_model.to(self.config.device)
        elif self.config.library == "diffusers":
            LOGGER.info("\t+ Loading Diffusion pipeline")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.config.model,
                device_map=self.config.device_map,
                **self.config.hub_kwargs,
                **self.automodel_kwargs,
            )
            if self.config.device_map is None:
                LOGGER.info(f"\t+ Moving pipeline to device: {self.config.device}")
                self.pretrained_model.to(self.config.device)
        elif self.is_bnb_quantized:
            LOGGER.info("\t+ Loading BnB quantized model")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.config.model,
                device_map=self.config.device_map,
                **self.config.hub_kwargs,
                **self.automodel_kwargs,
            )
        elif self.is_gptq_quantized or self.is_awq_quantized:
            LOGGER.info("\t+ Loading quantized model")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.config.model,
                # for gptq, we need to specify the device_map to either auto
                # or a cuda adevice to avoid any modules being assigned to cpu ¯\_(ツ)_/¯
                device_map=self.config.device_map or torch.device(self.config.device),
                **self.config.hub_kwargs,
                **self.automodel_kwargs,
            )
        elif self.config.device_map is not None:
            LOGGER.info(f"\t+ Loading model with device map: {self.config.device_map}")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.config.model,
                device_map=self.config.device_map,
                **self.config.hub_kwargs,
                **self.automodel_kwargs,
            )
        else:
            # this is the fastest way to load a model on a specific device
            # but not compatible with all quantization methods (and pipelines)
            LOGGER.info(f"\t+ Loading model directly on device: {self.config.device}")
            with torch.device(self.config.device):
                self.pretrained_model = self.automodel_class.from_pretrained(
                    pretrained_model_name_or_path=self.config.model, **self.config.hub_kwargs, **self.automodel_kwargs
                )

    def create_no_weights_model(self) -> None:
        LOGGER.info("\t+ Creating no weights model state_dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()

        if self.is_exllamav2:
            # for exllamav2 we need to add g_idx to the state_dict which
            # requires some information about linear layers dimensions
            with torch.device("meta"):
                meta_model = self.automodel_class.from_config(self.pretrained_config)
            for name, module in meta_model.named_modules():
                if hasattr(module, "in_features"):
                    state_dict[name + ".g_idx"] = torch.ones((module.in_features,), dtype=torch.int32)

        if self.is_quantized:
            # tricking from_pretrained to load the model as if it was quantized
            self.pretrained_config.quantization_config = self.quantization_config.to_dict()

        LOGGER.info("\t+ Creating no weights model directory")
        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights")
        os.makedirs(self.no_weights_model, exist_ok=True)

        LOGGER.info("\t+ Saving no weights model pretrained config")
        self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)

        LOGGER.info("\t+ Saving no weights model state_dict")
        save_file(
            filename=os.path.join(self.no_weights_model, "model.safetensors"),
            metadata={"format": "pt"},
            tensors=state_dict,
        )

    def load_model_with_no_weights(self) -> None:
        self.create_no_weights_model()

        with no_init_weights():
            original_model = self.config.model
            self.config.model = self.no_weights_model
            LOGGER.info("\t+ Loading no weights model")
            self.load_model_from_pretrained()
            self.config.model = original_model

        LOGGER.info("\t+ Randomizing model weights")
        randomize_weights(self.pretrained_model)
        LOGGER.info("\t+ Tying model weights")
        self.pretrained_model.tie_weights()

    def process_quantization_config(self) -> None:
        if self.is_gptq_quantized:
            LOGGER.info("\t+ Processing GPTQ config")
            from transformers import GPTQConfig

            self.quantization_config = GPTQConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        elif self.is_awq_quantized:
            LOGGER.info("\t+ Processing AWQ config")
            from transformers import AwqConfig

            self.quantization_config = AwqConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        elif self.is_bnb_quantized:
            LOGGER.info("\t+ Processing BitsAndBytes config")
            from transformers import BitsAndBytesConfig

            self.quantization_config = BitsAndBytesConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        else:
            self.quantization_config = None

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
        return (
            self.is_gptq_quantized
            and "exllama_config" in self.quantization_config
            and self.quantization_config["exllama_config"].get("version", None) == 2
        )

    @property
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.torch_dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)

        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation

        if self.config.low_cpu_mem_usage is not None:
            kwargs["low_cpu_mem_usage"] = self.config.low_cpu_mem_usage

        if self.is_quantized:
            kwargs["_fast_init"] = False
            kwargs["quantization_config"] = self.quantization_config

        return kwargs

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.library == "diffusers":
            return {"prompt": inputs["prompt"]}

        LOGGER.info(f"\t+ Moving inputs tensors to device {self.config.device}")
        for key, value in inputs.items():
            inputs[key] = value.to(self.config.device)

        if self.config.library == "timm":
            return {"x": inputs["pixel_values"]}

        return inputs

    @torch.inference_mode()
    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        with torch.autocast(device_type=self.config.device, dtype=self.amp_dtype, enabled=self.config.amp_autocast):
            return self.pretrained_model.forward(**inputs, **kwargs)

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
        LOGGER.info("\t+ Wrapping training arguments with transformers.TrainingArguments")
        training_arguments = TrainingArguments(**training_arguments)
        LOGGER.info("\t+ Wrapping model with transformers.Trainer")
        trainer = Trainer(
            args=training_arguments,
            model=self.pretrained_model,
            callbacks=training_callbacks,
            train_dataset=training_dataset,
            data_collator=training_data_collator,
        )
        LOGGER.info("\t+ Starting training")
        trainer.train()
        LOGGER.info("\t+ Training finished successfully")

    def seed(self):
        super().seed()
        torch.manual_seed(self.config.seed)

        if self.config.device == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            LOGGER.info("\t+ Cleaning backend temporary directory")
            self.tmpdir.cleanup()

        gc.collect()
