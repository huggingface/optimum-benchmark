from typing import Dict, Optional, Any, TYPE_CHECKING
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from logging import getLogger


import torch
import inspect
from torch import Tensor
from omegaconf import OmegaConf
from hydra.utils import get_class
from optimum.intel.openvino.utils import _HEAD_TO_AUTOMODELS
from openvino.runtime import __version__ as openvino_version
from optimum.intel import OVConfig as OVQuantizationConfig, OVQuantizer

if TYPE_CHECKING:
    from transformers.modeling_outputs import ModelOutput


from .base import Backend, BackendConfig
from .utils.openvino_utils import (
    DEFAULT_QUANTIZATION_CONFIG,
    DEFAULT_CALIBRATION_CONFIG,
)


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
    reshape: bool = False
    half: bool = False

    # quantization options
    quantization: bool = False
    quantization_config: Optional[Dict[str, Any]] = None

    # calibration options
    calibration: bool = True
    calibration_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        assert self.torch_dtype is None or self.torch_dtype == "float32", (
            "Only float32 is supported for torch_dtype in openvino backend. "
            f"Got {self.torch_dtype}"
        )

        if self.quantization:
            self.quantization_config = OmegaConf.merge(
                self.quantization_config or {},
                DEFAULT_QUANTIZATION_CONFIG,
            )

        if self.calibration:
            self.calibration_config = OmegaConf.merge(
                self.calibration_config or {},
                DEFAULT_CALIBRATION_CONFIG,
            )


class OVBackend(Backend):
    name: str = "openvino"
    config: OVConfig

    def __init__(
        self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(model, task, device, hub_kwargs)
        self.device = torch.device(device)

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
        self.config.torch_dtype = (
            getattr(torch, self.config.torch_dtype)
            if self.config.torch_dtype is not None
            else None
        )

        if self.config.quantization:
            self.config.quantization_config = OVQuantizationConfig(
                **self.config.quantization_config,
            )

        with TemporaryDirectory() as tmpdirname:
            if self.config.no_weights:
                raise NotImplementedError(
                    "no_weights is not supported for openvino backend"
                )
            else:
                self.load_model_from_pretrained()

            if self.config.quantization:
                self.quantize(tmpdirname)

    def load_model_from_pretrained(self) -> None:
        self.pretrained_model = self.ovmodel_class.from_pretrained(
            model_id=self.model,
            use_merged=self.config.use_merged,
            export=self.config.export,
            **self.hub_kwargs,
        )

    def quantize(self, tmpdirname: str) -> None:
        LOGGER.info("\t+ Attempting quantization")

        model = self.automodel_class.from_pretrained(self.model, **self.hub_kwargs)
        quantizer = OVQuantizer.from_pretrained(model)

        preprocess_class = get_class(self.config.calibration_config.preprocess_class)
        preprocess_function = preprocess_class(model_name_or_path=self.model)

        calibration_dataset = quantizer.get_calibration_dataset(
            dataset_name=self.config.calibration_config.dataset_name,
            num_samples=self.config.calibration_config.num_samples,
            dataset_config_name=self.config.calibration_config.dataset_config_name,
            dataset_split=self.config.calibration_config.dataset_split,
            preprocess_function=preprocess_function,
        )

        quantizer.quantize(
            calibration_dataset=calibration_dataset,
            save_directory=f"{tmpdirname}/quantized",
            quantization_config=self.config.quantization_config,
            # defaults
            batch_size=1,
            data_collator=None,
            remove_unused_columns=True,
            weights_only=False,
        )
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading quantized model")
        self.pretrained_model = self.ovmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/quantized",
            use_merged=self.config.use_merged,
        )

    def prepare_for_inference(self, input_shapes: Dict[str, int]) -> None:
        if self.config.reshape:
            static_shapes = {
                key: value
                for key, value in input_shapes.items()
                if key in inspect.getfullargspec(self.pretrained_model.reshape).args
            }
            LOGGER.info(f"\t+ Reshaping model with static shapes: {static_shapes}")
            self.pretrained_model.reshape(**static_shapes)

        if self.config.half:
            LOGGER.info("\t+ Converting model to half precision")
            self.pretrained_model.half()

        if self.config.reshape or self.config.half:
            LOGGER.info("\t+ Compiling model")
            self.pretrained_model.compile()

    def forward(self, input: Dict[str, Tensor], **kwargs) -> "ModelOutput":
        output = self.pretrained_model(**input, **kwargs)

        return output

    def generate(self, input: Dict[str, Tensor], **kwargs) -> "ModelOutput":
        output = self.pretrained_model.generate(**input, **kwargs)

        return output

    def train(self, **kwargs) -> None:
        pass
