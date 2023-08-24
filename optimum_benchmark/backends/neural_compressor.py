from typing import Dict, Optional, Any, TYPE_CHECKING
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from logging import getLogger

import torch
from torch import Tensor
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from optimum.intel.neural_compressor.quantization import INCQuantizer
from optimum.intel.neural_compressor.utils import _HEAD_TO_AUTOMODELS
from neural_compressor import __version__ as neural_compressor_version
from neural_compressor.config import (
    AccuracyCriterion,
    TuningCriterion,
    PostTrainingQuantConfig,
)

if TYPE_CHECKING:
    from transformers.utils import ModelOutput

from .base import Backend, BackendConfig
from .utils.neural_compressor_utils import (
    DEFAULT_QUANTIZATION_CONFIG,
    DEFAULT_CALIBRATION_CONFIG,
)


LOGGER = getLogger("neural_compressor")

OmegaConf.register_new_resolver("ptq_is_static", lambda approach: approach == "static")


@dataclass
class INCConfig(BackendConfig):
    name: str = "neural_compressor"
    version: str = neural_compressor_version
    _target_: str = "optimum_benchmark.backends.neural_compressor.INCBackend"

    # export options
    no_weights: bool = False

    # quantization options
    quantization: bool = False
    quantization_config: Optional[Dict[str, Any]] = None

    # calibration options
    calibration: bool = False
    calibration_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.no_weights:
            # TODO: implement no_weights for neural_compressor backend if possible
            raise NotImplementedError(
                "no_weights is not supported for neural_compressor backend"
            )

        if self.quantization:
            self.quantization_config = OmegaConf.merge(
                self.quantization_config if self.quantization_config else {},
                DEFAULT_QUANTIZATION_CONFIG,
            )
            if self.calibration_config["approach"] == "static":
                self.calibration = True

        if self.calibration:
            self.calibration_config = OmegaConf.merge(
                self.calibration_config if self.calibration_config else {},
                DEFAULT_CALIBRATION_CONFIG,
            )


class INCBackend(Backend):
    name: str = "neural_compressor"
    config: INCConfig

    def __init__(
        self, model: str, task: str, device: str, hub_kwargs: DictConfig
    ) -> None:
        super().__init__(model, task, device, hub_kwargs)
        self.device = torch.device(device)

        assert self.task in _HEAD_TO_AUTOMODELS, (
            f"INCBackend does not support task {self.task} yet. "
            f"Supported tasks are: {list(_HEAD_TO_AUTOMODELS.keys())}"
        )

        self.incmodel_class = get_class(
            f"optimum.intel.neural_compressor.{_HEAD_TO_AUTOMODELS[self.task]}"
        )
        LOGGER.info(
            f"\t+ Infered INCModel class {self.incmodel_class.__name__} "
            f"for task {self.task} and model_type {self.model_type}"
        )

    def configure(self, config: INCConfig) -> None:
        super().configure(config)

        if self.config.quantization:
            self.config.quantization_config["accuracy_criterion"] = AccuracyCriterion(
                **self.config.quantization_config["accuracy_criterion"]
            )
            self.config.quantization_config["tuning_criterion"] = TuningCriterion(
                **self.config.quantization_config["tuning_criterion"]
            )
            self.quantization_config = PostTrainingQuantConfig(
                **self.config.quantization_config
            )

        if self.config.calibration:
            self.config.calibration_config["preprocess_class"] = get_class(
                self.config.calibration_config["preprocess_class"]
            )
            self.config.calibration_config[
                "preprocess_function"
            ] = self.config.calibration_config["preprocess_class"](
                model_name_or_path=self.model
            )
            self.config.calibration_config.pop("preprocess_class")

        with TemporaryDirectory() as tmpdirname:
            if self.config.quantization:
                self.load_and_quantize_automodel(tmpdirname)
            else:
                self.load_incmodel()

    def load_and_quantize_automodel(self, tmpdirname: str) -> None:
        LOGGER.info("\t+ Loading pretrained AutoModel")
        model = self.automodel_class.from_pretrained(self.model, **self.hub_kwargs)
        LOGGER.info("\t+ Creating quantizer")
        quantizer = INCQuantizer.from_pretrained(
            model,
            eval_fn=None,
            calibration_fn=None,
            task=self.task,
        )

        if self.config.calibration:
            LOGGER.info("\t+ Loading calibration dataset")
            calibration_dataset = quantizer.get_calibration_dataset(
                **self.config.calibration_config
            )
        else:
            calibration_dataset = None

        LOGGER.info("\t+ Attempting quantization")
        quantizer.quantize(
            quantization_config=self.config.quantization_config,
            save_directory=f"{tmpdirname}/quantized",
            calibration_dataset=calibration_dataset,
            # default values
            batch_size=8,
            data_collator=None,
            remove_unused_columns=True,
            file_name=None,
        )

        LOGGER.info("\t+ Loading quantized INCModel")
        self.pretrained_model = self.incmodel_class.from_pretrained(
            model_name_or_path=f"{tmpdirname}/quantized",
        )

    def load_incmodel(self) -> None:
        if self.is_diffusion_pipeline():
            self.pretrained_model = self.incmodel_class.from_pretrained(
                model_name_or_path=self.model,
                **self.hub_kwargs,
            )
            self.pretrained_model.to(self.device)
        elif self.is_text_generation_model():
            self.pretrained_model = self.incmodel_class.from_pretrained(
                # for some reason only causalLM expects 
                # model_id instead of model_name_or_path
                model_id=self.model,
                device_map=self.device,
                **self.hub_kwargs,
            )
        else:
            self.pretrained_model = self.incmodel_class.from_pretrained(
                # for some reason only causalLM expects 
                # model_id instead of model_name_or_path
                model_name_or_path=self.model,
                device_map=self.device,
                **self.hub_kwargs,
            )

    def forward(self, input: Dict[str, Tensor], **kwargs) -> "ModelOutput":
        output = self.pretrained_model(**input, **kwargs)

        return output

    def generate(self, input: Dict[str, Tensor], **kwargs) -> "ModelOutput":
        output = self.pretrained_model.generate(**input, **kwargs)

        return output
