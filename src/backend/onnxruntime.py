from typing import Dict
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
    AutoOptimizationConfig,
    AutoQuantizationConfig,
)

from tempfile import TemporaryDirectory
from pandas import DataFrame
from torch import Tensor
import statistics
import gc

from src.backend.utils import json_to_df, split_data_across_runs, load_json
from src.backend.base import Backend, BackendConfig

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
    optimization: DictConfig = DictConfig({})
    # quantization options
    quantization: DictConfig = DictConfig({})


class ORTBackend(Backend):
    def configure(self, config: ORTConfig) -> None:
        LOGGER.info("Configuring onnxruntime backend")
        super().configure(config)

        session_options = SessionOptions()

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

        filtered_opt_config = {
            key: value
            for (key, value) in config.optimization.items()
            if value is not None
        }

        # always there since it's inferred from device
        for_gpu = filtered_opt_config.pop("for_gpu")
        print(filtered_opt_config)
        # at least one optimization parameter should be set
        if filtered_opt_config:
            LOGGER.info("Attempting to optimize model")
            optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)

            # optimization level for coherence with optimum's API
            level = filtered_opt_config.pop("level", None)
            if level is not None:
                filtered_opt_config["for_gpu"] = for_gpu
                LOGGER.info(f"\t+ Using onnxruntime optimization level: {level}")
                if filtered_opt_config:
                    LOGGER.info(f"\t+ Setting onnxruntime optimization parameters:")
                    for key, value in filtered_opt_config.items():
                        LOGGER.info(f"\t+ {key}: {value}")

                optimization_config = AutoOptimizationConfig.with_optimization_level(
                    optimization_level=level,
                    **filtered_opt_config,  # type: ignore
                )
            else:
                filtered_opt_config["optimize_for_gpu"] = for_gpu
                LOGGER.info("\t+ Setting onnxruntime optimization parameters:")
                for key, value in filtered_opt_config.items():
                    LOGGER.info(f"\t+ {key}: {value}")

                optimization_config = OptimizationConfig(**filtered_opt_config)  # type: ignore

            with TemporaryDirectory() as tmpdirname:
                LOGGER.info("\t+ Starting optimization")
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

        filtered_qnt_config = {
            key: value
            for (key, value) in config.quantization.items()
            if value is not None
        }

        if filtered_qnt_config:
            LOGGER.info("Attempting to quantize model")
            quantizer = ORTQuantizer.from_pretrained(self.pretrained_model)

            instrunctions = filtered_qnt_config.pop("instructions", None)

            if instrunctions is not None:
                LOGGER.info(
                    f"\t+ Using onnxruntime quantization instructions: {instrunctions}"
                )
                # well it should at least specify is_static (leaving validation for Pydatic)
                if filtered_qnt_config:
                    LOGGER.info(f"\t+ Setting onnxruntime quantization parameters:")
                    for key, value in filtered_qnt_config.items():
                        LOGGER.info(f"\t+ {key}: {value}")

                quantization_config = getattr(AutoQuantizationConfig, instrunctions)(
                    **filtered_qnt_config  # this is basically a leap of faith
                )

            else:
                # same as above
                if filtered_qnt_config:
                    LOGGER.info("\t+ Setting onnxruntime quantization parameters:")
                    for key, value in filtered_qnt_config.items():
                        LOGGER.info(f"\t+ {key}: {value}")

                    # should be handeled by Pydantic
                    if filtered_qnt_config.get("format", None) is not None:
                        filtered_qnt_config["format"] = QuantFormat.from_string(
                            filtered_qnt_config["format"]
                        )
                    if filtered_qnt_config.get("mode", None) is not None:
                        filtered_qnt_config["mode"] = QuantizationMode.from_string(
                            filtered_qnt_config["mode"]
                        )
                    if filtered_qnt_config.get("activations_dtype", None) is not None:
                        filtered_qnt_config["activations_dtype"] = QuantType.from_string(
                            filtered_qnt_config["activations_dtype"]
                        )
                    if filtered_qnt_config.get("weights_dtype", None) is not None:
                        filtered_qnt_config["weights_dtype"] = QuantType.from_string(
                            filtered_qnt_config["weights_dtype"]
                        )

                quantization_config = QuantizationConfig(**filtered_qnt_config)  # type: ignore

            with TemporaryDirectory() as tmpdirname:
                LOGGER.info("\t+ Starting quantization")
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
        gc.collect()