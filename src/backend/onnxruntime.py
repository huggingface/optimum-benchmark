from logging import getLogger
from typing import Dict, Optional
from dataclasses import dataclass
from omegaconf.dictconfig import DictConfig

from optimum.onnxruntime.configuration import AutoOptimizationConfig, AutoQuantizationConfig
from onnxruntime import SessionOptions, __version__ as ort_version # type: ignore
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from optimum.pipelines import ORT_SUPPORTED_TASKS
from pandas import DataFrame, read_json
from tempfile import TemporaryDirectory
from torch import Tensor

from src.backend.base import Backend, BackendConfig

BACKEND_NAME = 'onnxruntime'

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = ort_version

    # basic options
    provider: str = 'CPUExecutionProvider'
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
        LOGGER.info("Configuring onnxruntime backend:")
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

        try:
            ortmodel_class = ORT_SUPPORTED_TASKS[self.task]['class'][0]
        except KeyError:
            raise NotImplementedError(
                f"Feature {self.task} not supported by onnxruntime backend")

        LOGGER.info(
            f"\t+ Loading model {self.model} for task {self.task} on {self.device}")
        self.pretrained_model = ortmodel_class.from_pretrained(
            model_id=self.model,
            provider=config.provider,
            use_io_binding=config.use_io_binding,
            session_options=session_options,
            export=True,
        )

        if config.optimization_level is not None:
            LOGGER.info("\t+ Optimizing model")
            optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)

            custom_opt_config = {
                key: value
                for (key, value)
                in config.optimization_parameters.items()
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
                **custom_opt_config # type: ignore
            )

            with TemporaryDirectory() as tmpdirname:
                optimizer.optimize(
                    save_dir=f'{tmpdirname}/{self.model}.onnx',
                    optimization_config=optimization_config,
                )
                self.pretrained_model = ortmodel_class.from_pretrained(
                    model_id=f'{tmpdirname}/{self.model}.onnx',
                    session_options=session_options,
                    use_io_binding=config.use_io_binding,
                    provider=config.provider,
                )

        if config.quantization_strategy is not None:
            LOGGER.info("\t+ Quantizing model")
            quantizer = ORTQuantizer.from_pretrained(self.pretrained_model)

            custom_qnt_config = {
                key: value
                for (key, value)
                in config.quantization_parameters.items()
                if value is not None
            }

            LOGGER.info(
                f"\t+ Setting onnxruntime quantization strategy with "
                f"backend.quantization_strategy({config.quantization_strategy})"
                f"and overriding quantization config with custom "
                f"backend.quantization_parameters({custom_qnt_config})"
            )

            quantization_class = getattr(AutoQuantizationConfig, config.quantization_strategy)
            quantization_config = quantization_class(**custom_qnt_config)

            with TemporaryDirectory() as tmpdirname:
                quantizer.quantize(
                    save_dir=f'{tmpdirname}/{self.model}.onnx',
                    quantization_config=quantization_config,
                )
                self.pretrained_model = ortmodel_class.from_pretrained(
                    model_id=f'{tmpdirname}/{self.model}.onnx',
                    session_options=session_options,
                    use_io_binding=config.use_io_binding,
                    provider=config.provider,
                )

    def run_profiling(self, inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int) -> DataFrame:
        LOGGER.info("Can't warmup onnxruntime model before profiling")

        LOGGER.info("Profiling model")
        latencies = []
        while (sum(latencies) < benchmark_duration):
            latency = self.inference_latency(inputs)
            latencies.append(latency)

        profiling_file = self.pretrained_model.model.end_profiling() # type: ignore
        profiling_results = read_json(profiling_file, orient='records')

        profiling_results = profiling_results[profiling_results['cat'] == 'Node']
        profiling_results = profiling_results[['name', 'args', 'dur']]
        profiling_results['dur'] = (
            profiling_results['dur'] / 1e6) / len(latencies)

        profiling_results.rename(
            columns={
                'name': 'Node name',
                'args': 'to be processed (gather nodes before and after)',
                'dur': 'Node latency mean (s)',
            },
            inplace=True,
        )
        # can't get std from onnxruntime
        profiling_results['Node latency std (s)'] = 0

        return profiling_results
