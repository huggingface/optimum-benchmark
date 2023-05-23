from logging import getLogger
from typing import Dict, Optional
from dataclasses import dataclass
from omegaconf.dictconfig import DictConfig


from optimum.onnxruntime import ORTOptimizer, ORTQuantizer, ORTModel
from optimum.pipelines import ORT_SUPPORTED_TASKS
from optimum.onnxruntime.configuration import (
    AutoOptimizationConfig,
    AutoQuantizationConfig,
)
from tempfile import TemporaryDirectory
from pandas import DataFrame
from torch import Tensor
import onnxruntime
import statistics

from src.backend.utils import json_to_df, split_data_across_runs, load_json
from src.backend.base import Backend, BackendConfig

BACKEND_NAME = "onnxruntime"

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = onnxruntime.__version__

    # basic options
    provider: str = "CPUExecutionProvider"
    use_io_binding: bool = False
    enable_profiling: bool = False

    # graph optimization options
    optimization_level: Optional[str] = None
    optimization_parameters: DictConfig = DictConfig({})
    # auto quantization options
    quantization_strategy: Optional[str] = None
    quantization_parameters: DictConfig = DictConfig({})


class ORTBackend(Backend):
    def configure(self, config: ORTConfig) -> None:
        LOGGER.info("Configuring onnxruntime backend")
        super().configure(config)

        session_options = onnxruntime.SessionOptions()

        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime intra_op_num_threads({config.intra_op_num_threads})"
            )
            session_options.intra_op_num_threads = config.intra_op_num_threads

        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime inter_op_num_threads({config.inter_op_num_threads})"
            )
            session_options.inter_op_num_threads = config.inter_op_num_threads

        if config.enable_profiling:
            LOGGER.info("\t+ Enabling onnxruntime profiling")
            session_options.enable_profiling = True

        ortmodel_class = ORT_SUPPORTED_TASKS[self.task]["class"][0]

        LOGGER.info(f"Loading model {self.model} with {ortmodel_class.__name__}")
        self.pretrained_model: ORTModel = ortmodel_class.from_pretrained(
            model_id=self.model,
            provider=config.provider,
            use_io_binding=config.use_io_binding,
            session_options=session_options,
            export=True,
        )

        if config.optimization_level is not None:
            LOGGER.info("Optimizing model")
            optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)

            custom_opt_config = {
                key: value
                for (key, value) in config.optimization_parameters.items()
                if value is not None
            }

            LOGGER.info(
                f"\t+ Setting onnxruntime optimization level with "
                f"backend.optimization_level({config.optimization_level}) "
                f"and overriding optimization config with custom "
                f"backend.optimization_parameters({custom_opt_config})"
            )
            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=config.optimization_level,
                **custom_opt_config,  # type: ignore
            )

            with TemporaryDirectory() as tmpdirname:
                optimizer.optimize(
                    save_dir=f"{tmpdirname}/{self.model}.onnx",
                    optimization_config=optimization_config,
                )
                LOGGER.info("\t+ Loading optimized model")
                self.pretrained_model = ortmodel_class.from_pretrained(
                    model_id=f"{tmpdirname}/{self.model}.onnx",
                    session_options=session_options,
                    use_io_binding=config.use_io_binding,
                    provider=config.provider,
                )

        if config.quantization_strategy is not None:
            LOGGER.info("Quantizing model")
            quantizer = ORTQuantizer.from_pretrained(self.pretrained_model)

            custom_qnt_config = {
                key: value
                for (key, value) in config.quantization_parameters.items()
                if value is not None
            }

            LOGGER.info(
                f"\t+ Setting onnxruntime quantization strategy with "
                f"backend.quantization_strategy({config.quantization_strategy})"
                f"and overriding quantization config with custom "
                f"backend.quantization_parameters({custom_qnt_config})"
            )

            quantization_class = getattr(
                AutoQuantizationConfig, config.quantization_strategy
            )
            quantization_config = quantization_class(**custom_qnt_config)

            with TemporaryDirectory() as tmpdirname:
                quantizer.quantize(
                    save_dir=f"{tmpdirname}/{self.model}.onnx",
                    quantization_config=quantization_config,
                )

                LOGGER.info("\t+ Loading quantized model")
                self.pretrained_model = ortmodel_class.from_pretrained(
                    model_id=f"{tmpdirname}/{self.model}.onnx",
                    session_options=session_options,
                    use_io_binding=config.use_io_binding,
                    provider=config.provider,
                )

    def run_inference(
        self, dummy_inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int
    ) -> DataFrame:
        LOGGER.info("Running backend inference")

        LOGGER.info("\t+ Warming up the model")
        for _ in range(warmup_runs):
            self.pretrained_model(**dummy_inputs)

        LOGGER.info("\t+ Tracking inference latency")
        inference_latencies = []
        while sum(inference_latencies) < benchmark_duration:
            latency = self.track_inference_latency(dummy_inputs)
            inference_latencies.append(latency)

        LOGGER.info("\t+ Processing inference results")
        inference_results = DataFrame(
            {
                "Model latency mean (s)": statistics.mean(inference_latencies),
                "Model latency median (s)": statistics.median(inference_latencies),
                "Model latency stdev (s)": statistics.stdev(inference_latencies)
                if len(inference_latencies) > 1
                else 0,
                "Model Throughput (s^-1)": len(inference_latencies)
                / benchmark_duration,
            },
            index=[0],
        )

        return inference_results

    def run_profiling(
        self, dummy_inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int
    ) -> DataFrame:
        LOGGER.info("Running backend profiling")

        LOGGER.info("\t+ Warming up the model")
        for _ in range(warmup_runs):
            self.pretrained_model(**dummy_inputs)

        LOGGER.info("\t+ Profiling the model")
        profiling_latencies = []
        while sum(profiling_latencies) < benchmark_duration:
            latency = self.track_inference_latency(dummy_inputs)
            profiling_latencies.append(latency)

        profiling_file = self.pretrained_model.model.end_profiling()  # type: ignore

        LOGGER.info("\t+ Parsing profiling results")
        profiling_json = load_json(profiling_file)
        profiling_json, num_runs = split_data_across_runs(
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
