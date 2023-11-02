import inspect
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict

from hydra.utils import get_class
from openvino.runtime import properties
from optimum.intel.openvino import OVConfig as OVQuantizationConfig  # naming conflict
from optimum.intel.openvino import OVQuantizer

from ..base import Backend
from .config import OVConfig
from .utils import TASKS_TO_OVMODEL

LOGGER = getLogger("openvino")


class OVBackend(Backend[OVConfig]):
    NAME: str = "openvino"

    def __init__(self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]) -> None:
        super().__init__(model, task, device, hub_kwargs)
        self.validate_device()
        self.validate_task()

        self.ovmodel_class = get_class(TASKS_TO_OVMODEL[self.task])
        ortmodel_name = self.ovmodel_class.__name__
        LOGGER.info(f"Inferred OVModel class {ortmodel_name} for task {self.task} and model_type {self.model_type}")

    def validate_task(self) -> None:
        if self.task not in TASKS_TO_OVMODEL:
            raise NotImplementedError(f"OVBackend does not support task {self.task}")

    def validate_device(self) -> None:
        if self.device.type != "cpu":
            raise ValueError(f"OVBackend only supports CPU devices, got {self.device.type}")

    def configure(self, config: OVConfig) -> None:
        super().configure(config)

        self.openvino_config = self.config.openvino_config.copy()
        if self.config.inter_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting inter_op_num_threads to {self.config.inter_op_num_threads}")
            self.openvino_config[properties.inference_num_threads()] = self.config.inter_op_num_threads

        if self.config.intra_op_num_threads is not None:
            raise NotImplementedError("OVBackend does not support intra_op_num_threads")

        self.tmpdir = TemporaryDirectory()

        if self.config.quantization:
            self.load_automodel()
            self.quantize_automodel()
            self.delete_pretrained_model()  # deletes automodel
            self.export = False  # quantized model is already exported
        else:
            self.export = self.config.export  # to not change the config's values

        self.load_ovmodel()
        self.tmpdir.cleanup()

    def load_automodel(self) -> None:
        self.pretrained_model = self.automodel_class.from_pretrained(self.model, **self.hub_kwargs)

    @property
    def ovmodel_kwargs(self) -> Dict[str, Any]:
        if self.is_text_generation_model():
            return {"use_cache": self.config.use_cache, "use_merged": self.config.use_merged}
        else:
            return {}

    def load_ovmodel(self) -> None:
        self.pretrained_model = self.ovmodel_class.from_pretrained(
            self.model,
            export=self.export,
            ov_config=self.openvino_config,
            **self.ovmodel_kwargs,
            **self.hub_kwargs,
        )

    def quantize_automodel(self) -> None:
        LOGGER.info("\t+ Attempting quantization")
        quantized_model_path = f"{self.tmpdir.name}/quantized"
        LOGGER.info("\t+ Processing quantization config")
        quantization_config = OVQuantizationConfig(**self.config.quantization_config)
        LOGGER.info("\t+ Creating quantizer")
        quantizer = OVQuantizer.from_pretrained(self.pretrained_model, task=self.task, seed=self.config.seed)
        LOGGER.info("\t+ Processing calibration config")
        calibration_config = self.config.calibration_config.copy()
        preprocess_class = get_class(calibration_config.pop("preprocess_class"))
        calibration_config["preprocess_function"] = preprocess_class(model_name_or_path=self.model)
        LOGGER.info("\t+ Loading calibration dataset")
        calibration_dataset = quantizer.get_calibration_dataset(**calibration_config)
        LOGGER.info("\t+ Quantizing model")
        quantizer.quantize(
            quantization_config=quantization_config,
            save_directory=quantized_model_path,
            calibration_dataset=calibration_dataset,
            # TODO: add support for these
            remove_unused_columns=True,
            data_collator=None,
            weights_only=False,
            file_name=None,
            batch_size=1,
        )
        self.model = quantized_model_path

    def prepare_for_inference(self, **kwargs) -> None:
        input_shapes = kwargs["input_shapes"]
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

    def clean(self) -> None:
        super().clean()
        if hasattr(self, "tmpdir"):
            self.tmpdir.cleanup()
