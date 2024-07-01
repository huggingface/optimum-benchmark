import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from safetensors.torch import save_file
from vllm import LLM, SamplingParams

from ...task_utils import TEXT_GENERATION_TASKS
from ..base import Backend
from ..transformers_utils import random_init_weights
from .config import VLLMConfig


class VLLMBackend(Backend[VLLMConfig]):
    NAME: str = "vllm"

    def __init__(self, config: VLLMConfig) -> None:
        super().__init__(config)
        self.validate_task()

        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            self.logger.info("\t+ Loading no weights model")
            self.load_model_with_no_weights()
        else:
            self.logger.info("\t+ Downloading pretrained model")
            self.download_pretrained_model()

            self.logger.info("\t+ Preparing generation config")
            self.prepare_generation_config()

            self.logger.info("\t+ Loading pretrained model")
            self.load_model_from_pretrained()

        self.tmpdir.cleanup()

    def download_pretrained_model(self) -> None:
        with torch.device("meta"):
            self.automodel_class.from_pretrained(self.config.model, **self.config.model_kwargs)

    def prepare_generation_config(self) -> None:
        self.generation_config.eos_token_id = None
        self.generation_config.pad_token_id = None

        model_cache_folder = f"models/{self.config.model}".replace("/", "--")
        model_cache_path = f"{HUGGINGFACE_HUB_CACHE}/{model_cache_folder}"
        snapshot_file = f"{model_cache_path}/refs/{self.config.model_kwargs.get('revision', 'main')}"
        snapshot_ref = open(snapshot_file, "r").read().strip()
        model_snapshot_path = f"{model_cache_path}/snapshots/{snapshot_ref}"
        self.logger.info("\t+ Saving new pretrained generation config")
        self.generation_config.save_pretrained(save_directory=model_snapshot_path)

    def create_no_weights_model(self) -> None:
        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights_model")
        self.logger.info("\t+ Creating no weights model directory")
        os.makedirs(self.no_weights_model, exist_ok=True)
        self.logger.info("\t+ Creating no weights model state dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()
        self.logger.info("\t+ Saving no weights model safetensors")
        safetensor = os.path.join(self.no_weights_model, "model.safetensors")
        save_file(tensors=state_dict, filename=safetensor, metadata={"format": "pt"})
        self.logger.info("\t+ Saving no weights model pretrained config")
        self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)
        self.logger.info("\t+ Saving no weights model pretrained processor")
        self.pretrained_processor.save_pretrained(save_directory=self.no_weights_model)
        # unlike Transformers, vLLM won't accept any missing tensors so we need to materialize the model
        self.logger.info(f"\t+ Loading no weights model from {self.no_weights_model}")
        with random_init_weights():
            self.pretrained_model = self.automodel_class.from_pretrained(
                self.no_weights_model, **self.config.model_kwargs, device_map="auto", _fast_init=False
            )
        self.logger.info("\t+ Saving no weights model")
        self.pretrained_model.save_pretrained(save_directory=self.no_weights_model)
        del self.pretrained_model
        torch.cuda.empty_cache()

        if self.config.task in TEXT_GENERATION_TASKS:
            self.logger.info("\t+ Modifying generation config for fixed length generation")
            self.generation_config.eos_token_id = None
            self.generation_config.pad_token_id = None

            self.logger.info("\t+ Saving new pretrained generation config")
            self.generation_config.save_pretrained(save_directory=self.no_weights_model)

    def load_model_with_no_weights(self) -> None:
        self.logger.info("\t+ Creating no weights model")
        self.create_no_weights_model()

        original_model, self.config.model = self.config.model, self.no_weights_model
        self.logger.info("\t+ Loading no weights model")
        self.load_model_from_pretrained()
        self.config.model = original_model

    def load_model_from_pretrained(self) -> None:
        self.pretrained_model = LLM(
            model=self.config.model,
            # tokenizer
            tokenizer=self.config.processor,
            tokenizer_mode=self.config.tokenizer_mode,
            skip_tokenizer_init=self.config.skip_tokenizer_init,
            # device
            device=self.config.device,
            # parallelism
            tensor_parallel_size=self.config.tensor_parallel_size,
            # precision
            quantization=self.config.quantization,
            dtype=self.config.dtype,
            # memory
            swap_space=self.config.swap_space,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            # cuda graphs
            enforce_eager=self.config.enforce_eager,
            max_context_len_to_capture=self.config.max_context_len_to_capture,
            max_seq_len_to_capture=self.config.max_seq_len_to_capture,
            # kernels
            disable_custom_all_reduce=self.config.disable_custom_all_reduce,
            # additional stuff
            trust_remote_code=self.config.model_kwargs.get("trust_remote_code", False),
            tokenizer_revision=self.config.processor_kwargs.get("revision", None),
            revision=self.config.model_kwargs.get("revision", None),
            seed=self.config.seed,
        )

    def validate_task(self) -> None:
        if self.config.task not in ["text-generation"]:
            raise ValueError(f"Task {self.config.task} not supported by {self.NAME}")

    def prepare_inputs(
        self, inputs: Dict[str, Any], input_shapes: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, input_shapes = super().prepare_inputs(inputs, input_shapes)

        if self.config.task in TEXT_GENERATION_TASKS:
            inputs = {"prompts": self.pretrained_processor.batch_decode(inputs["input_ids"])}
        else:
            raise NotImplementedError(f"vLLM does not support task {self.config.task}")

        return inputs, input_shapes

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
        return self.pretrained_model.generate(
            **inputs,
            use_tqdm=False,
            sampling_params=SamplingParams(
                ignore_eos=True,
                detokenize=True,
                seed=self.config.seed,
                n=kwargs.get("num_return_sequences"),
                max_tokens=kwargs.get("max_new_tokens"),
                min_tokens=kwargs.get("min_new_tokens"),
                use_beam_search=kwargs.get("num_beams") > 1,
                logits_processors=kwargs.get("logits_processors", None),
            ),
        )

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return self.pretrained_model.generate(
            **inputs,
            use_tqdm=False,
            sampling_params=SamplingParams(
                ignore_eos=True,
                seed=self.config.seed,
                n=kwargs.get("num_return_sequences"),
                max_tokens=kwargs.get("max_new_tokens"),
                min_tokens=kwargs.get("min_new_tokens"),
                use_beam_search=kwargs.get("num_beams") > 1,
                logits_processors=kwargs.get("logits_processors", None),
            ),
        )

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
        return self.pretrained_model.generate(
            **inputs,
            use_tqdm=False,
            sampling_params=SamplingParams(
                ignore_eos=True,
                n=kwargs.get("num_return_sequences"),
                max_tokens=kwargs.get("max_new_tokens"),
                min_tokens=kwargs.get("min_new_tokens"),
                use_beam_search=kwargs.get("num_beams") > 1,
                logits_processors=kwargs.get("logits_processors", None),
            ),
        )
