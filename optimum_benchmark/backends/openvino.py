import torch
import inspect
from torch import Tensor
from logging import getLogger
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import get_class
from typing import Dict, Optional
from tempfile import TemporaryDirectory

try:
    from openvino.runtime import __version__ as openvino_version
except ImportError:
    openvino_version = "Not installed"

from optimum_benchmark.backends.base import Backend, BackendConfig

LOGGER = getLogger("openvino")


@dataclass
class OVConfig(BackendConfig):
    name: str = "openvino"
    version: str = openvino_version
    _target_: str = "optimum_benchmark.backends.openvino.OVBackend"

    # export options
    export: bool = True
    no_weights: bool = False
    use_merged: Optional[bool] = None
    torch_dtype: Optional[str] = None

    # compiling options
    dynamic_shapes: bool = True
    reshape: bool = False
    half: bool = False

    # quantization options
    quantization: bool = False
    quantization_config: DictConfig = DictConfig(
        {
            "compression": None,
            "input_info": None,
            "save_onnx_model": False,
        }
    )

    # calibration options
    calibration_config: DictConfig = DictConfig(
        {
            "dataset_name": "glue",
            "num_samples": 300,
            "dataset_config_name": "sst2",
            "dataset_split": "train",
            "preprocess_batch": True,
            "preprocess_class": "optimum_benchmark.preprocessors.glue.GluePreprocessor",
        }
    )


class OVBackend(Backend):
    def __init__(
        self, model: str, task: str, device: str, hub_kwargs: DictConfig
    ) -> None:
        super().__init__(model, task, device, hub_kwargs)

        from optimum.intel.openvino.utils import _HEAD_TO_AUTOMODELS

        self.ovmodel_class = get_class(
            f"optimum.intel.openvino.{_HEAD_TO_AUTOMODELS[self.task]}"
        )

        LOGGER.info(
            f"\t+ Infered OVModel class {self.ovmodel_class.__name__} "
            f"for task {self.task} and model_type {self.model_type}"
        )

    def configure(self, config: OVConfig) -> None:
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
                    "no_weights is not supported for openvino backend"
                )
            else:
                self.load_model_from_pretrained(config)

            if config.quantization:
                self.quantize(config, tmpdirname)

        self.reshape = config.reshape
        if self.reshape:
            LOGGER.info("\t+ Model input will be reshaped and compiled")

        self.half = config.half
        if self.half:
            LOGGER.info("\t+ Model will be converted to half precision and compiled")

    def load_model_from_pretrained(self, config: OVConfig) -> None:
        if self.torch_dtype is not None and self.torch_dtype != torch.float32:
            raise NotImplementedError(
                "Loading from pretrained is only supported with torch_dtype float32 for now"
            )
        self.pretrained_model = self.ovmodel_class.from_pretrained(
            model_id=self.model,
            use_merged=config.use_merged,
            export=config.export,
            **self.hub_kwargs,
        )

    def quantize(self, config: OVConfig, tmpdirname: str) -> None:
        LOGGER.info("\t+ Attempting quantization")

        from optimum.intel import OVConfig as OVQuantizationConfig, OVQuantizer

        model = self.automodel_class.from_pretrained(self.model, **self.hub_kwargs)
        quantizer = OVQuantizer.from_pretrained(model)
        quantization_config = OVQuantizationConfig(
            **config.quantization_config,
        )

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
            save_directory=f"{tmpdirname}/quantized",
            quantization_config=quantization_config,
            calibration_dataset=calibration_dataset,
        )
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading quantized model")
        self.pretrained_model = self.ovmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/quantized",
        )

    def prepare_for_forward(self, input_shapes: Dict[str, Tensor]) -> None:
        if self.reshape:
            relevant_shapes = {
                k: v
                for k, v in input_shapes.items()
                if k
                in inspect.signature(self.pretrained_model.reshape).parameters.keys()
            }
            LOGGER.info(f"\t+ Reshaping model with input shapes {relevant_shapes}")
            self.pretrained_model.reshape(**relevant_shapes)

        if self.half:
            LOGGER.info(f"\t+ Converting model to half precision")
            self.pretrained_model.half()

        if self.reshape or self.half:
            LOGGER.info(f"\t+ Compiling model")
            self.pretrained_model.compile()

    def forward(self, input: Dict[str, Tensor], **kwargs) -> Tensor:
        output = self.pretrained_model(**input, **kwargs)[0]

        return output

    def generate(self, input: Dict[str, Tensor], **kwargs) -> Tensor:
        output = self.pretrained_model.generate(**input, **kwargs)[0]

        return output

    def train(self, **kwargs) -> None:
        pass
