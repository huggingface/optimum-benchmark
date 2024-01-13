import gc
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict

from hydra.utils import get_class
from neural_compressor.config import (
    AccuracyCriterion,
    PostTrainingQuantConfig,
    TuningCriterion,
)
from optimum.intel.neural_compressor.quantization import INCQuantizer

from ..base import Backend
from .config import INCConfig
from .utils import TASKS_TO_INCMODELS

LOGGER = getLogger("neural-compressor")


class INCBackend(Backend[INCConfig]):
    NAME: str = "neural-compressor"

    def __init__(self, model: str, task: str, library: str, device: str, hub_kwargs: Dict[str, Any]) -> None:
        super().__init__(model, task, library, device, hub_kwargs)
        self.validate_device()
        self.validate_task()

        self.incmodel_class = get_class(TASKS_TO_INCMODELS[self.task])
        LOGGER.info(
            f"Inferred INCModel {self.incmodel_class.__name__} for task {self.task} and model_type {self.model_type}"
        )

    def validate_device(self) -> None:
        if self.device != "cpu":
            raise ValueError(f"INCBackend only supports CPU devices, got {self.device}")

    def validate_task(self) -> None:
        if self.task not in TASKS_TO_INCMODELS:
            raise NotImplementedError(f"INCBackend does not support task {self.task}")

    def configure(self, config: INCConfig) -> None:
        super().configure(config)

        self.tmpdir = TemporaryDirectory()

        if self.config.ptq_quantization:
            self.load_automodel_from_pretrained()
            self.quantize_automodel()
            self.delete_pretrained_model()

        self.load_incmodel_from_pretrained()

    def load_automodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading AutoModel")
        self.pretrained_model = self.automodel_class.from_pretrained(self.model, **self.hub_kwargs)

    def load_incmodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading INCModel")
        self.pretrained_model = self.incmodel_class.from_pretrained(self.model, **self.hub_kwargs)

    def quantize_automodel(self) -> None:
        LOGGER.info("\t+ Attempting to quantize model")
        quantized_model_path = f"{self.tmpdir.name}/quantized"
        LOGGER.info("\t+ Processing quantization config")
        ptq_quantization_config = self.config.ptq_quantization_config.copy()
        ptq_quantization_config["accuracy_criterion"] = AccuracyCriterion(
            **ptq_quantization_config["accuracy_criterion"]
        )
        ptq_quantization_config["tuning_criterion"] = TuningCriterion(**ptq_quantization_config["tuning_criterion"])
        ptq_quantization_config = PostTrainingQuantConfig(**ptq_quantization_config)
        LOGGER.info("\t+ Creating quantizer")
        quantizer = INCQuantizer.from_pretrained(
            self.pretrained_model,
            task=self.task,
            seed=self.config.seed,
            # TODO: add support for these
            eval_fn=None,
            calibration_fn=None,
        )

        if self.config.calibration:
            LOGGER.info("\t+ Processing calibration config")
            calibration_config = self.config.calibration_config.copy()
            preprocess_class = get_class(calibration_config.pop("preprocess_class"))
            calibration_config["preprocess_function"] = preprocess_class(model_name_or_path=self.model)
            LOGGER.info("\t+ Loading calibration dataset")
            calibration_dataset = quantizer.get_calibration_dataset(**calibration_config)
        else:
            calibration_dataset = None

        LOGGER.info("\t+ Quantizing model")
        quantizer.quantize(
            quantization_config=ptq_quantization_config,
            save_directory=quantized_model_path,
            calibration_dataset=calibration_dataset,
            # TODO: add support for these
            remove_unused_columns=True,
            data_collator=None,
            file_name=None,
            batch_size=8,
        )
        self.model = quantized_model_path

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.library == "diffusers":
            return {"prompt": inputs["prompt"]}

        return inputs

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            self.tmpdir.cleanup()

        gc.collect()
