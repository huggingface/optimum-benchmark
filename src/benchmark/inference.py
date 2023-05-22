from dataclasses import dataclass
from logging import getLogger
from typing import Dict


import torch
from torch import Tensor
from pandas import DataFrame
from transformers import AutoConfig
from optimum.exporters import TasksManager

from src.backend.base import Backend
from src.benchmark.base import Benchmark, BenchmarkConfig

BENCHMARK_NAME = "inference"
LOGGER = getLogger(BENCHMARK_NAME)


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = BENCHMARK_NAME

    warmup_runs: int = 5
    benchmark_duration: int = 5


class InferenceBenchmark(Benchmark):
    def configure(self, config: InferenceConfig):
        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

    def run(self, backend: Backend) -> None:
        dummy_inputs = self.generate_dummy_inputs()
        self.inference_results = backend.run_inference(
            dummy_inputs, self.warmup_runs, self.benchmark_duration
        )

    @property
    def results(self) -> DataFrame:
        return self.inference_results

    def save(self, path: str = "") -> None:
        LOGGER.info("Saving inference results")
        self.inference_results.to_csv(path + "inference_results.csv")

    def generate_dummy_inputs(self) -> Dict[str, Tensor]:
        LOGGER.info(f"Generating dummy inputs")

        auto_config = AutoConfig.from_pretrained(self.model)
        model_type = auto_config.model_type
        LOGGER.info(f"\t+ Using {model_type} as model type")

        onnx_config = TasksManager._SUPPORTED_MODEL_TYPE[model_type]["onnx"][self.task](
            auto_config
        )
        LOGGER.info(f"\t+ Using {onnx_config.__class__.__name__} as onnx config")

        input_names = list(onnx_config.inputs.keys())  # type: ignore
        LOGGER.info(f"\t+ Using {input_names} as model input names")

        dummy_inputs = dict()
        for input_name in input_names:
            dummy_input_generator = None

            for dummy_input_generator_class in onnx_config.DUMMY_INPUT_GENERATOR_CLASSES:  # type: ignore
                if input_name in dummy_input_generator_class.SUPPORTED_INPUT_NAMES:  # type: ignore
                    dummy_input_generator = dummy_input_generator_class(
                        task=self.task,
                        normalized_config=onnx_config.NORMALIZED_CONFIG_CLASS(  # type: ignore
                            auto_config
                        ),
                    )

            if dummy_input_generator is None:
                raise ValueError(
                    f"Could not find dummy input generator for {input_name}"
                )

            LOGGER.info(
                f"\t+ Generating dummy input for {input_name} using {dummy_input_generator.__class__.__name__}"
            )

            dummy_inputs[input_name] = dummy_input_generator.generate(
                input_name, framework="pt"
            ).to(self.device)

            if input_name == "attention_mask" and "input_ids" in dummy_inputs:
                # this is for bettertransformer
                dummy_inputs["attention_mask"] = torch.ones_like(
                    dummy_inputs["input_ids"]
                )

        return dummy_inputs
