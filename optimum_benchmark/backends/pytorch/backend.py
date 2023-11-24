import gc
import os
from logging import getLogger
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from transformers import TrainerCallback, TrainerState
from transformers.utils import ModelOutput

from ..base import Backend
from .config import PyTorchConfig
from .utils import DTYPES_MAPPING, randomize_weights, TransformersDataParallel

# bachend logger
LOGGER = getLogger("pytorch")


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME: str = "pytorch"

    def __init__(self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]):
        super().__init__(model, task, device, hub_kwargs)

    def configure(self, config: PyTorchConfig) -> None:
        super().configure(config)

        automodel = self.automodel_class.__name__
        LOGGER.info(f"\t+ Inferred AutoModel class {automodel} for task {self.task} and model_type {self.model_type}")

        # for now we rely on this env variable to know if we're in a distributed setting
        if os.environ.get("LOCAL_WORLD_SIZE", None) is not None:
            LOGGER.info(f"\t+ Detected local world size: {os.environ['LOCAL_WORLD_SIZE']}")
            local_rank = int(os.environ["LOCAL_RANK"])
            LOGGER.info(f"\t+ Detected local rank: {local_rank}")
            available_devices = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
            LOGGER.info(f"\t+ Detected available devices: {available_devices}")
            default_device = available_devices[local_rank]
            LOGGER.info(f"\t+ Setting default device to: {default_device}")
            torch.cuda.set_device(default_device)

        # Gradients options
        if self.config.disable_grad:
            LOGGER.info("\t+ Disabling gradients")
            torch.set_grad_enabled(False)

        # Threading options
        if self.config.inter_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting pytorch inter_op_num_threads({self.config.inter_op_num_threads}))")
            torch.set_num_threads(self.config.inter_op_num_threads)
        if self.config.intra_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting pytorch intra_op_num_threads({self.config.intra_op_num_threads}))")
            torch.set_num_interop_threads(self.config.intra_op_num_threads)

        # Dtypes options
        self.torch_dtype = getattr(torch, self.config.torch_dtype) if self.config.torch_dtype is not None else None
        self.amp_dtype = getattr(torch, self.config.amp_dtype) if self.config.amp_dtype is not None else None

        # Load model
        if self.config.no_weights:
            self.load_model_from_config()
        else:
            self.load_model_from_pretrained()

        # Eval mode
        if self.config.eval_mode:
            if self.is_diffusion_pipeline():
                LOGGER.info("\t+ Diffusion pipeline are always in eval mode")
            else:
                LOGGER.info("\t+ Turning on model's eval mode")
                self.pretrained_model.eval()

        # BetterTransformer
        if self.config.to_bettertransformer:
            LOGGER.info("\t+ Enabling BetterTransformer")
            self.pretrained_model.to_bettertransformer()

        # Compile model
        if self.config.torch_compile:
            if self.is_diffusion_pipeline():
                LOGGER.info("\t+ Using torch.compile on unet forward pass")
                # TODO: should we compile vae and/or clip as well ?
                self.pretrained_model.unet.forward = torch.compile(
                    self.pretrained_model.unet.forward,
                    **self.config.torch_compile_config,
                )
            else:
                LOGGER.info("\t+ Using torch.compile on forward pass")
                self.pretrained_model.forward = torch.compile(
                    self.pretrained_model.forward,
                    **self.config.torch_compile_config,
                )

        if self.config.peft_strategy is not None:
            LOGGER.info("\t+ Applying PEFT")
            from peft import get_peft_model

            from ..peft_utils import get_peft_config_class

            peft_config_class = get_peft_config_class(self.config.peft_strategy)
            peft_config = peft_config_class(**self.config.peft_config)
            self.pretrained_model = get_peft_model(self.pretrained_model, peft_config=peft_config)

        if self.config.deepspeed_inference:
            LOGGER.info("\t+ Using DeepSpeed-Inference")
            from deepspeed import init_inference

            self.pretrained_model = init_inference(
                self.pretrained_model,
                config=self.config.deepspeed_inference_config,
                dtype=self.torch_dtype,
            )

        if self.config.data_parallel:
            LOGGER.info("\t+ Using TransformersDataParallel")
            self.pretrained_model = TransformersDataParallel(self.pretrained_model)

    def load_model_from_pretrained(self) -> None:
        # iniline quantization or quantization config modification
        if self.config.quantization_scheme == "gptq":
            LOGGER.info("\t+ Processing GPTQ config")
            from transformers import GPTQConfig

            self.quantization_config = GPTQConfig(**self.config.quantization_config)
        elif self.config.quantization_scheme == "awq":
            LOGGER.info("\t+ Processing AWQ config")
            from transformers import AwqConfig

            self.quantization_config = AwqConfig(**self.config.quantization_config)
        elif self.config.quantization_scheme == "bnb":
            LOGGER.info("\t+ Processing BitsAndBytes config")
            from transformers import BitsAndBytesConfig

            self.quantization_config = self.config.quantization_config.copy()
            if self.quantization_config.get("bnb_4bit_compute_dtype", None) is not None:
                self.quantization_config["bnb_4bit_compute_dtype"] = getattr(
                    torch, self.quantization_config["bnb_4bit_compute_dtype"]
                )
            self.quantization_config = BitsAndBytesConfig(**self.quantization_config)
        else:
            self.quantization_config = None

        if self.is_diffusion_pipeline():
            LOGGER.info("\t+ Loading diffusion pipeline")
            self.pretrained_model = self.automodel_class.from_pretrained(
                self.model,
                torch_dtype=self.torch_dtype,
                device_map=self.config.device_map,
                **self.hub_kwargs,
            )
            if self.config.device_map is None:
                LOGGER.info(f"\t+ Moving diffusion pipeline to device: {self.device}")
                self.pretrained_model.to(self.device)

        elif self.config.device_map is not None:
            LOGGER.info(f"\t+ Loading model with device_map: {self.config.device_map}")
            self.pretrained_model = self.automodel_class.from_pretrained(
                self.model,
                torch_dtype=self.torch_dtype,
                device_map=self.config.device_map,
                **self.automodel_kwargs,
                **self.hub_kwargs,
            )
        elif self.is_quantized():
            LOGGER.info(f"\t+ Loading quantized model and moving it to device: {self.device}")
            self.pretrained_model = self.automodel_class.from_pretrained(
                self.model,
                torch_dtype=self.torch_dtype,
                **self.automodel_kwargs,
                **self.hub_kwargs,
            ).to(self.device)
        else:
            LOGGER.info(f"\t+ Loading model directly on device: {self.device}")
            with torch.device(self.device):
                # this is extremely faster than the above method
                self.pretrained_model = self.automodel_class.from_pretrained(
                    self.model,
                    torch_dtype=self.torch_dtype,
                    **self.automodel_kwargs,
                    **self.hub_kwargs,
                )

    def is_quantized(self) -> bool:
        return self.config.quantization_scheme is not None or hasattr(self.pretrained_config, "quantization_config")

    def is_gptq_model(self) -> bool:
        return self.config.quantization_scheme == "gptq" or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method", None) == "gptq"
        )

    @property
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.is_quantized():
            if self.is_gptq_model():
                kwargs["device_map"] = torch.device(self.device)
            else:
                kwargs["low_cpu_mem_usage"] = True

        if self.quantization_config is not None:
            kwargs["quantization_config"] = self.quantization_config

        if self.config.use_flash_attention_2:
            kwargs["use_flash_attention_2"] = True

        return kwargs

    def load_model_from_config(self) -> None:
        from accelerate import init_empty_weights

        LOGGER.info("\t+ Initializing empty weights model on device: meta")
        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_config(
                config=self.pretrained_config,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.hub_kwargs.get("trust_remote_code", False),
            )

        if self.config.quantization_scheme is not None:
            if self.config.quantization_scheme == "bnb":
                from accelerate.utils import (
                    BnbQuantizationConfig,
                    load_and_quantize_model,
                )

                LOGGER.info("\t+ Materializing model on cpu for quantization to not OOM")
                self.pretrained_model.to_empty(device="cpu")
                LOGGER.info("\t+ Randomizing model weights")
                randomize_weights(self.pretrained_model)
                LOGGER.info("\t+ Processing BnBQuantizationConfig")
                bnb_quantization_config = BnbQuantizationConfig(
                    **self.config.quantization_config,
                    torch_dtype=DTYPES_MAPPING[self.config.torch_dtype],
                    keep_in_fp32_modules=getattr(self.pretrained_model, "keep_in_fp32_modules", None),
                )
                LOGGER.info("\t+ Quantizing model while on cpu")
                self.pretrained_model = load_and_quantize_model(self.pretrained_model, bnb_quantization_config)
                LOGGER.info(f"\t+ Moving model to target device: {self.device}")
                self.pretrained_model.to(device=self.device)
            elif self.config.quantization_scheme == "gptq":
                LOGGER.info("\t+ Processing GPTQ config")
                from optimum.gptq import GPTQQuantizer
                from transformers import GPTQConfig

                LOGGER.info("\t+ Materializing model on cpu for quantization to not OOM")
                self.pretrained_model.to_empty(device="cpu")
                LOGGER.info("\t+ Randomizing model weights")
                randomize_weights(self.pretrained_model)
                LOGGER.info("\t+ Creating GPTQQuantizer")
                gptq_quantizer = GPTQQuantizer(**self.config.quantization_config)
                LOGGER.info("\t+ Quantizing model while on cpu")
                gptq_quantizer.convert_model(self.pretrained_model)
                LOGGER.info(f"\t+ Moving model to target device: {self.device}")
                self.pretrained_model.to(device=self.device)
                LOGGER.info("\t+ Postprocessing model")
                self.pretrained_model.config.quantization_config = GPTQConfig.from_dict(gptq_quantizer.to_dict())
                self.pretrained_model._is_quantized_training_enabled = True
                gptq_quantizer.post_init_model(self.pretrained_model)
            else:
                raise ValueError("Only bnb and gptq quantization schemes are supported with no_weights=True")
        else:
            LOGGER.info(f"\t+ Materializing model on device: {self.device}")
            self.pretrained_model.to_empty(device=self.device)
            LOGGER.info("\t+ Randomizing model weights")
            randomize_weights(self.pretrained_model)

        LOGGER.info("\t+ Tying weights")
        self.pretrained_model.tie_weights()

    def forward(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> "ModelOutput":
        if self.is_diffusion_pipeline():
            return super().forward(input, kwargs)
        elif self.config.amp_autocast:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                return super().forward(input, kwargs)
        else:
            return super().forward(input, kwargs)

    def generate(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> "ModelOutput":
        if self.is_diffusion_pipeline():
            return super().generate(input, kwargs)
        elif self.config.amp_autocast:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                return super().generate(input, kwargs)
        else:
            return super().generate(input, kwargs)

    def train(
        self,
        training_dataset: "Dataset",
        training_arguments: Dict[str, Any],
        training_callbacks: List["TrainerCallback"],
        training_data_collator: Callable,
        dataset_format: str = "torch",
    ) -> "TrainerState":
        from transformers import Trainer, TrainingArguments

        LOGGER.info(f"\t+ Setting dataset format to `{dataset_format}`.")
        training_dataset.set_format(type=dataset_format, columns=list(training_dataset.features.keys()))
        LOGGER.info("\t+ Wrapping training arguments with transformers.TrainingArguments")
        training_arguments = TrainingArguments(**training_arguments)
        LOGGER.info("\t+ Wrapping model with transformers.Trainer")
        trainer = Trainer(
            model=self.pretrained_model,
            args=training_arguments,
            callbacks=training_callbacks,
            train_dataset=training_dataset,
            data_collator=training_data_collator,
        )
        LOGGER.info("\t+ Starting training")
        trainer.train()
        LOGGER.info("\t+ Training finished successfully")

        return trainer.state

    def seed(self):
        super().seed()
        torch.manual_seed(self.config.seed)

        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

    def clean(self) -> None:
        super().clean()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        gc.collect()
