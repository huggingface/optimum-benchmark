import gc
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import TrainerCallback, TrainerState
from transformers.utils import ModelOutput
from transformers.utils.logging import set_verbosity_error

from ..base import Backend
from .config import PyTorchConfig
from .utils import TransformersDataParallel, randomize_weights

# bachend logger
LOGGER = getLogger("pytorch")

# disable transformers logging
set_verbosity_error()


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME: str = "pytorch"

    def __init__(self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]):
        super().__init__(model, task, device, hub_kwargs)

    def configure(self, config: PyTorchConfig) -> None:
        super().configure(config)

        automodel = self.automodel_class.__name__
        LOGGER.info(f"\t+ Inferred AutoModel class {automodel} for task {self.task} and model_type {self.model_type}")

        # for now we rely on this env variable to know if we're in a distributed setting
        if self.is_distributed():
            LOGGER.info("\t+ Detected distributed cuda environment")
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            LOGGER.info(f"\t+ Detected local world size: {local_world_size}")
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            LOGGER.info(f"\t+ Detected cuda visible devices: {visible_devices}")

            assert local_world_size == len(visible_devices) == torch.cuda.device_count(), (
                f"LOCAL_WORLD_SIZE ({local_world_size}) and CUDA_VISIBLE_DEVICES ({visible_devices}) "
                f"and torch.cuda.device_count() ({torch.cuda.device_count()}), "
                "are not consistent with each other."
            )

            local_rank = int(os.environ["LOCAL_RANK"])
            LOGGER.info(f"\t+ Detected local rank: {local_rank}")

            LOGGER.info("\t+ Setting default cuda device to local rank")
            torch.cuda.set_device(local_rank)

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
        self.amp_dtype = getattr(torch, self.config.amp_dtype) if self.config.amp_dtype is not None else None

        if self.is_quantized():
            # iniline quantization or quantization config modification
            LOGGER.info("\t+ Processing quantization config")
            self.process_quantization_config()
        else:
            self.quantization_config = None

        # Load model
        if self.config.no_weights and self.is_diffusion_pipeline():
            raise ValueError("Diffusion Pipelines are not supported with no_weights=True")
        if self.config.no_weights:
            LOGGER.info("\t+ Loading model with no weights")
            self.load_model_with_no_weights()
        else:
            LOGGER.info("\t+ Loading model with pretrained weights")
            self.load_model_from_pretrained()

        # Eval mode
        if self.config.eval_mode:
            if self.is_diffusion_pipeline():
                LOGGER.info("\t+ Diffusion pipeline is in eval mode")
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
                dtype=getattr(self.pretrained_model, "dtype", None),
            )

        if self.config.data_parallel:
            LOGGER.info("\t+ Using TransformersDataParallel")
            self.pretrained_model = TransformersDataParallel(self.pretrained_model)

    def load_model_from_pretrained(self) -> None:
        if self.is_diffusion_pipeline():
            LOGGER.info("\t+ Loading pipeline")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                device_map=self.config.device_map,
                **self.automodel_kwargs,
                **self.hub_kwargs,
            )
            if self.config.device_map is None:
                LOGGER.info(f"\t+ Moving pipeline to device: {self.device}")
                self.pretrained_model.to(self.device)
        elif self.is_bnb_quantized():
            LOGGER.info("\t+ Loading BnB quantized model")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                device_map=self.config.device_map,
                # this avoids unnecessary memory usage when loading quantized models
                low_cpu_mem_usage=True,
                **self.automodel_kwargs,
                **self.hub_kwargs,
            )
        elif self.is_gptq_quantized() or self.is_awq_quantized():
            LOGGER.info("\t+ Loading quantized model")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                # for gptq, we need to specify the device_map to either auto
                # or a cuda adevice to avoid any modules being assigned to cpu
                device_map=self.config.device_map or torch.device(self.device),
                # this avoids unnecessary memory usage when loading quantized models
                low_cpu_mem_usage=True,
                **self.automodel_kwargs,
                **self.hub_kwargs,
            )
        elif self.config.device_map is not None:
            LOGGER.info(f"\t+ Loading model with device map: {self.config.device_map}")
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                device_map=self.config.device_map,
                **self.automodel_kwargs,
                **self.hub_kwargs,
            )
        else:
            # this is the fastest way to load a model on a specific device
            LOGGER.info(f"\t+ Loading model directly on device: {self.device}")
            with torch.device(self.device):
                self.pretrained_model = self.automodel_class.from_pretrained(
                    pretrained_model_name_or_path=self.model,
                    **self.automodel_kwargs,
                    **self.hub_kwargs,
                )

    def load_model_with_no_weights(self) -> None:
        self.tmp_dir = TemporaryDirectory()

        original_model = self.model
        no_weights_model = os.path.join(self.tmp_dir.name, "no_weights")

        LOGGER.info("\t+ Creating no weights model directory")
        if not os.path.exists(no_weights_model):
            os.makedirs(no_weights_model)

        if self.is_quantized():
            # tricking from_pretrained to load the model as if it was quantized
            self.pretrained_config.quantization_config = self.quantization_config.to_dict()

        LOGGER.info(f"\t+ Saving pretrained config to {no_weights_model}")
        self.pretrained_config.save_pretrained(save_directory=no_weights_model)

        LOGGER.info(f"\t+ Creating no weights model to {no_weights_model}")
        state_dict = torch.nn.Linear(1, 1).state_dict()

        if self.is_exllamav2():
            # for exllamav2 we need to add g_idx to the state_dict
            LOGGER.info("\t+ Loading meta model")
            with torch.device("meta"):
                meta_model = self.automodel_class.from_config(self.pretrained_config)

            LOGGER.info("\t+ Setting g_idx for ExllamaV2")
            for name, module in meta_model.named_modules():
                # loading to exllama v2's QuantLinear creates g_idx with bad values
                if hasattr(module, "in_features"):
                    state_dict[name + ".g_idx"] = torch.ones((module.in_features,), dtype=torch.int32)

        LOGGER.info(f"\t+ Saving no weights model to {no_weights_model}")
        save_file(
            filename=os.path.join(no_weights_model, "model.safetensors"),
            metadata={"format": "pt"},
            tensors=state_dict,
        )

        LOGGER.info("\t+ Loading no weights model")
        self.model = no_weights_model
        self.load_model_from_pretrained()
        self.model = original_model

        if not self.is_quantized():
            # TODO: verify if this can be extended to quantized models
            # (not sure how torch.Tensor.normal_ works on quantized tensors)
            LOGGER.info("\t+ Randomizing model weights")
            randomize_weights(self.pretrained_model)
            LOGGER.info("\t+ Tying model weights after randomization")
            self.pretrained_model.tie_weights()

    def process_quantization_config(self) -> None:
        if self.is_gptq_quantized():
            LOGGER.info("\t+ Processing GPTQ config")
            from transformers import GPTQConfig

            self.quantization_config = GPTQConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        elif self.is_awq_quantized():
            LOGGER.info("\t+ Processing AWQ config")
            from transformers import AwqConfig

            self.quantization_config = AwqConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        elif self.is_bnb_quantized():
            LOGGER.info("\t+ Processing BitsAndBytes config")
            from transformers import BitsAndBytesConfig

            self.quantization_config = BitsAndBytesConfig(
                **dict(getattr(self.pretrained_config, "quantization_config", {}), **self.config.quantization_config)
            )
        else:
            self.quantization_config = None

    def is_distributed(self) -> bool:
        return os.environ.get("WORLD_SIZE", None) is not None

    def is_quantized(self) -> bool:
        return self.config.quantization_scheme is not None or hasattr(self.pretrained_config, "quantization_config")

    def is_bnb_quantized(self) -> bool:
        return self.config.quantization_scheme == "bnb" or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method", None) == "bnb"
        )

    def is_gptq_quantized(self) -> bool:
        return self.config.quantization_scheme == "gptq" or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method", None) == "gptq"
        )

    def is_awq_quantized(self) -> bool:
        return self.config.quantization_scheme == "awq" or (
            hasattr(self.pretrained_config, "quantization_config")
            and self.pretrained_config.quantization_config.get("quant_method", None) == "awq"
        )

    def is_exllamav2(self) -> bool:
        return (
            self.is_quantized()
            and self.is_gptq_quantized()
            and "exllama_config" in self.config.quantization_config
            and self.config.quantization_config["exllama_config"]["version"] == 2
        )

    @property
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.torch_dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)

        if self.config.use_flash_attention_2:
            kwargs["use_flash_attention_2"] = True

        if self.is_gptq_quantized() or self.is_bnb_quantized():
            # awq quantization doesn't support overriding the quantization
            # config by passing quantization_config to from_pretrained
            kwargs["quantization_config"] = self.quantization_config

        if self.config.no_weights:
            # when no_weights=True, the state_dict is empty so from_pretrained will try to randomly
            # initialize every missing weights, we don't want that, so we set fast_init to False
            kwargs["_fast_init"] = False

        return kwargs

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
        training_dataset: Dataset,
        training_arguments: Dict[str, Any],
        training_callbacks: List[TrainerCallback],
        training_data_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
    ) -> TrainerState:
        from transformers import Trainer, TrainingArguments

        LOGGER.info("\t+ Setting dataset format to `torch`")
        training_dataset.set_format(type="torch", columns=list(training_dataset.features.keys()))
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
            LOGGER.info("\t+ Emptying CUDA cache")
            torch.cuda.empty_cache()

        if hasattr(self, "tmp_dir"):
            LOGGER.info("\t+ Cleaning temporary directory")
            self.tmp_dir.cleanup()

        gc.collect()
