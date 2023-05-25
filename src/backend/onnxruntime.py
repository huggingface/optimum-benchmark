import os
from typing import Dict, List
from logging import getLogger
from dataclasses import dataclass
from omegaconf.dictconfig import DictConfig

from onnxruntime import __version__ as onnxruntime_version, SessionOptions  # type: ignore
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer, ORTModel
from optimum.pipelines import ORT_SUPPORTED_TASKS
from optimum.onnxruntime.configuration import (
    OptimizationConfig,
    QuantizationConfig,
    # AutoOptimizationConfig,
    # AutoQuantizationConfig,
)

from tempfile import TemporaryDirectory
from pandas import DataFrame
from torch import Tensor
import gc

from src.backend.base import Backend, BackendConfig
from src.tracker import Tracker
from src.utils import (
    json_to_df,
    split_data_across_runs,
    load_json,
)

BACKEND_NAME = "onnxruntime"

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = onnxruntime_version

    # basic options
    provider: str = "CPUExecutionProvider"
    use_io_binding: bool = False
    enable_profiling: bool = False

    # optimization options
    enable_optimization: bool = False
    optimization: DictConfig = DictConfig({})

    # quantization options
    enable_quantization: bool = False
    quantization: DictConfig = DictConfig({})


class ORTBackend(Backend):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)
        self.ortmodel_class = ORT_SUPPORTED_TASKS[self.task]["class"][0]

    def configure(self, config: ORTConfig) -> None:
        LOGGER.info("Configuring onnxruntime backend")
        super().configure(config)

        self.session_options = SessionOptions()

        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime intra_op_num_threads({config.intra_op_num_threads})"
            )
            self.session_options.intra_op_num_threads = config.intra_op_num_threads

        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime inter_op_num_threads({config.inter_op_num_threads})"
            )
            self.session_options.inter_op_num_threads = config.inter_op_num_threads

        if config.enable_profiling:
            LOGGER.info("\t+ Enabling onnxruntime profiling")
            self.session_options.enable_profiling = True

        LOGGER.info(f"Loading model {self.model} with {self.ortmodel_class.__name__}")
        self.pretrained_model: ORTModel = self.ortmodel_class.from_pretrained(
            model_id=self.model,
            session_options=self.session_options,
            provider=config.provider,
            use_io_binding=config.use_io_binding,
            export=True,
        )

        with TemporaryDirectory() as tmpdirname:
            if config.enable_optimization:
                self.optimize_model(config, tmpdirname)

            if config.enable_quantization:
                self.quantize_model(config, tmpdirname)

    def optimize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        LOGGER.info("\t+ Setting onnxruntime optimization parameters:")
        for key, value in config.optimization.items():
            LOGGER.info(f"\t+ {key}: {value}")

        optimization_config = OptimizationConfig(**config.optimization)  # type: ignore

        LOGGER.info("\t+ Starting optimization")
        optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)
        optimizer.optimize(
            save_dir=f"{tmpdirname}/optimized",
            optimization_config=optimization_config,
        )
        LOGGER.info("\t+ Loading optimized model")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/optimized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
        )

    def quantize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        LOGGER.info("\t+ Setting onnxruntime quantization parameters:")
        for key, value in config.quantization.items():
            LOGGER.info(f"\t+ {key}: {value}")

        # should be handeled by Pydantic
        if config.quantization.get("format", None) is not None:
            config.quantization["format"] = QuantFormat.from_string(
                config.quantization["format"]
            )
        if config.quantization.get("mode", None) is not None:
            config.quantization["mode"] = QuantizationMode.from_string(
                config.quantization["mode"]
            )
        if config.quantization.get("activations_dtype", None) is not None:
            config.quantization["activations_dtype"] = QuantType.from_string(
                config.quantization["activations_dtype"]
            )
        if config.quantization.get("weights_dtype", None) is not None:
            config.quantization["weights_dtype"] = QuantType.from_string(
                config.quantization["weights_dtype"]
            )

        quantization_config = QuantizationConfig(**config.quantization)  # type: ignore

        LOGGER.info("\t+ Starting quantization")
        model_dir = self.pretrained_model.model_save_dir
        components = [file for file in os.listdir(model_dir) if file.endswith(".onnx")]
        for component in components:
            LOGGER.info(f"\t+ Quantizing {component}")
            quantizer = ORTQuantizer.from_pretrained(model_dir, file_name=component)
            quantizer.quantize(
                save_dir=f"{tmpdirname}/quantized",
                quantization_config=quantization_config,
            )

        LOGGER.info("\t+ Loading quantized model")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/quantized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
        )

    def run_inference(
        self, dummy_inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int
    ) -> List[float]:
        LOGGER.info("Running inference on onnxruntime backend")
        LOGGER.info("\t+ Warming up the model")
        for _ in range(warmup_runs):
            self.pretrained_model(**dummy_inputs)

        LOGGER.info("\t+ Tracking inference latency")
        tracker = Tracker(device=self.device)
        while sum(tracker.tracked_latencies) < benchmark_duration:
            with tracker.track_latency():
                self.pretrained_model(**dummy_inputs)

        inference_latencies = tracker.tracked_latencies

        return inference_latencies

    def run_profiling(
        self, dummy_inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int
    ) -> DataFrame:
        LOGGER.info("Running backend profiling")

        LOGGER.info("\t+ Warming up the model")
        for _ in range(warmup_runs):
            self.pretrained_model(**dummy_inputs)

        LOGGER.info("\t+ Profiling the model")
        tracker = Tracker(device=self.device)
        while sum(tracker.tracked_latencies) < benchmark_duration:
            with tracker.track_latency():
                self.pretrained_model(**dummy_inputs)

        LOGGER.info("\t+ Parsing profiling results")
        profiling_file = self.pretrained_model.model.end_profiling()  # type: ignore
        profiling_json = load_json(profiling_file)
        profiling_json, _ = split_data_across_runs(
            profiling_json, start=warmup_runs, end=None
        )
        profiling_df = json_to_df(profiling_json)
        profiling_results = (
            profiling_df.groupby(["Kernel name", "Op name"])
            .aggregate(
                {
                    "Kernel mean latency (us)": "mean",
                    "Kernel median latency (us)": "median",
                    "Kernel stdev latency (us)": "std",
                },
                set_axis=False,
            )
            .reset_index()
        )

        return profiling_results

    def clean(self) -> None:
        LOGGER.info("Cleaning onnxruntime backend")
        del self.pretrained_model
        gc.collect()
