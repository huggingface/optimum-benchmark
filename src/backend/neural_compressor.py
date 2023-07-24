import torch
from torch import Tensor
import neural_compressor
from logging import getLogger
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_class
from dataclasses import dataclass
from typing import Dict, Optional
from tempfile import TemporaryDirectory


import optimum.intel.neural_compressor as optimum_neural_compressor
from optimum.intel.neural_compressor.utils import _HEAD_TO_AUTOMODELS
from optimum.intel.neural_compressor.quantization import INCQuantizer
from neural_compressor.config import (
    AccuracyCriterion,
    TuningCriterion,
    PostTrainingQuantConfig,
)

from src.backend.base import Backend, BackendConfig

OmegaConf.register_new_resolver(
    "ptq_approach_is_static",
    lambda approach: approach == "static",
)


LOGGER = getLogger("neural_compressor")


@dataclass
class INCConfig(BackendConfig):
    name: str = "neural_compressor"
    version: str = neural_compressor.__version__
    _target_: str = "src.backend.neural_compressor.INCBackend"

    # export options
    no_weights: bool = False
    torch_dtype: Optional[str] = None

    # quantization options
    quantization: bool = False
    quantization_config: DictConfig = DictConfig(
        {
            "device": "cpu",
            "backend": "default",
            "domain": "auto",
            "recipes": {},
            "quant_format": "default",
            "inputs": [],
            "outputs": [],
            "approach": "static",
            "calibration_sampling_size": [100],
            "op_type_dict": None,
            "op_name_dict": None,
            "reduce_range": None,
            "example_inputs": None,
            "excluded_precisions": [],
            "quant_level": "auto",
            "accuracy_criterion": DictConfig(
                {
                    "higher_is_better": True,
                    "criterion": "relative",
                    "tolerable_loss": 0.01,
                }
            ),
            "tuning_criterion": DictConfig(
                {
                    "strategy": "basic",
                    "strategy_kwargs": None,
                    "timeout": 0,
                    "max_trials": 100,
                    "objective": "performance",
                }
            ),
            "diagnosis": False,
        }
    )

    # calibration options
    calibration: bool = "${ptq_approach_is_static:${backend.quantization_config.approach}}"  # type: ignore
    calibration_config: DictConfig = DictConfig(
        {
            "dataset_name": "glue",
            "num_samples": 300,
            "dataset_config_name": "sst2",
            "dataset_split": "train",
            "preprocess_batch": True,
            "preprocess_class": "src.preprocessors.glue.GluePreprocessor",
        }
    )


class INCBackend(Backend):
    def __init__(
        self, model: str, task: str, device: str, cache_kwargs: DictConfig
    ) -> None:
        super().__init__(model, task, device, cache_kwargs)

        self.incmodel_class = getattr(
            optimum_neural_compressor, _HEAD_TO_AUTOMODELS[self.task]
        )

        LOGGER.info(
            f"\t+ Infered INCModel class {self.incmodel_class.__name__} "
            f"for task {self.task} and model_type {self.pretrained_config.model_type}"
        )

    def configure(self, config: INCConfig) -> None:
        super().configure(config)

        # Set torch dtype
        self.torch_dtype = (
            getattr(torch, config.torch_dtype)  # in case of torch.dtype
            if config.torch_dtype is not None and hasattr(torch, config.torch_dtype)
            else None  # in case of string or None
        )
        LOGGER.info(
            f"\t+ Using torch dtype({self.torch_dtype}) for weights loading and export"
        )

        with TemporaryDirectory() as tmpdirname:
            if config.no_weights:
                raise NotImplementedError(
                    "no_weights is not supported for neural_compressor backend"
                )
            else:
                self.load_model_from_pretrained(config)

            if config.quantization:
                self.quantize_model(config, tmpdirname)

    def load_model_from_pretrained(self, config: INCConfig) -> None:
        if self.torch_dtype is not None and self.torch_dtype != torch.float32:
            raise NotImplementedError(
                "Loading from pretrained is only supported with torch_dtype float32 for now"
            )
        self.pretrained_model = self.incmodel_class.from_pretrained(
            model_name_or_path=self.model,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            **self.cache_kwargs,
        )

    def quantize_model(self, config: INCConfig, tmpdirname: str) -> None:
        LOGGER.info("\t+ Attempting quantization")

        quantization_config = OmegaConf.to_container(config.quantization_config)
        quantization_config["accuracy_criterion"] = AccuracyCriterion(
            **config.quantization_config.accuracy_criterion
        )
        quantization_config["tuning_criterion"] = TuningCriterion(
            **config.quantization_config.tuning_criterion
        )
        quantization_config = PostTrainingQuantConfig(**quantization_config)

        model = self.automodel_class.from_pretrained(self.model, **self.cache_kwargs)
        quantizer = INCQuantizer.from_pretrained(model, task=self.task)

        if config.calibration:
            preprocess_class = get_class(config.calibration_config.preprocess_class)
            preprocess_function = preprocess_class(model_name_or_path=self.model)

            calibration_dataset = quantizer.get_calibration_dataset(
                dataset_name=config.calibration_config.dataset_name,
                num_samples=config.calibration_config.num_samples,
                dataset_config_name=config.calibration_config.dataset_config_name,
                dataset_split=config.calibration_config.dataset_split,
                preprocess_function=preprocess_function,
            )

        quantizer.quantize(
            save_onnx_model=False,
            quantization_config=quantization_config,
            calibration_dataset=calibration_dataset,
            save_directory=f"{tmpdirname}/quantized",
        )

        self.delete_pretrained_model()
        LOGGER.info("\t+ Loading quantized model")
        self.pretrained_model = self.incmodel_class.from_pretrained(
            model_name_or_path=f"{tmpdirname}/quantized",
        )

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        output = self.pretrained_model(**input)[0]

        return output

    def generate(self, input: Dict[str, Tensor], new_tokens: int) -> Tensor:
        output = self.pretrained_model.generate(
            **input,
            pad_token_id=self.pretrained_model.config.eos_token_id,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            num_beams=1,
        )[0]

        return output
