import os
import traceback
from abc import ABC, abstractmethod
from itertools import product
from logging import getLogger
from typing import Any, Dict, List, Optional

from llm_perf.common.utils import (
    CANONICAL_PRETRAINED_OPEN_LLM_LIST,
    OPEN_LLM_LIST,
    PRETRAINED_OPEN_LLM_LIST,
)
from optimum_benchmark import Benchmark, BenchmarkConfig, BenchmarkReport
from optimum_benchmark.logging_utils import setup_logging


class BenchmarkRunner(ABC):
    def __init__(self, backend: str, hardware: str, subset: Optional[str] = None, machine: Optional[str] = None):
        self.backend = backend
        self.hardware = hardware
        self.subset = subset or os.getenv("SUBSET", None)
        self.machine = machine or os.getenv("MACHINE", None)
        self.logger = getLogger("llm-perf-backend")

        if self.machine is None and self.subset is None:
            self.push_repo_id = f"optimum-benchmark/llm-perf-{self.backend}-{self.hardware}-debug"
            self.canonical_pretrained_open_llm_list = ["gpt2"]
            self.subset = "unquantized"
        elif self.machine is not None and self.subset is not None:
            self.push_repo_id = (
                f"optimum-benchmark/llm-perf-{self.backend}-{self.hardware}-{self.subset}-{self.machine}"
            )
        else:
            raise ValueError("Either both MACHINE and SUBSET should be set for benchmarking or neither for debugging")

        self.attention_configs = self._get_attention_configs()
        self.weights_configs = self._get_weights_configs(self.subset)

        self.logger.info(f"len(OPEN_LLM_LIST): {len(OPEN_LLM_LIST)}")
        self.logger.info(f"len(PRETRAINED_OPEN_LLM_LIST): {len(PRETRAINED_OPEN_LLM_LIST)}")
        self.logger.info(f"len(CANONICAL_PRETRAINED_OPEN_LLM_LIST): {len(CANONICAL_PRETRAINED_OPEN_LLM_LIST)}")

    @abstractmethod
    def _get_weights_configs(self, subset: str) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError("This method should be implemented in the child class")

    @abstractmethod
    def _get_attention_configs(self) -> List[str]:
        raise NotImplementedError("This method should be implemented in the child class")

    @abstractmethod
    def is_benchmark_supported(self, weights_config: str, attn_implementation: str) -> bool:
        raise NotImplementedError("This method should be implemented in the child class")

    def run_benchmarks(self):
        os.environ["LOG_TO_FILE"] = "0"
        os.environ["LOG_LEVEL"] = "INFO"
        setup_logging(level="INFO", prefix="MAIN-PROCESS")

        models_attentions_weights = list(
            product(CANONICAL_PRETRAINED_OPEN_LLM_LIST, self.attention_configs, self.weights_configs.keys())
        )

        self.logger.info(
            f"Running a total of {len(models_attentions_weights)} benchmarks, "
            f"with {len(CANONICAL_PRETRAINED_OPEN_LLM_LIST)} models, "
            f"{len(self.attention_configs)} attentions implementations "
            f"and {len(self.weights_configs)} weights configurations."
        )

        for model, attn_implementation, weights_config in models_attentions_weights:
            self.run_benchmark(model, attn_implementation, weights_config)

    def is_benchmark_conducted(self, push_repo_id, subfolder):
        try:
            report = BenchmarkReport.from_pretrained(repo_id=push_repo_id, subfolder=subfolder)
            if "traceback" in report.to_dict():
                return False
            else:
                return True
        except Exception:
            return False

    def run_benchmark(self, model: str, attn_implementation: str, weights_config: str):
        benchmark_name = f"{weights_config}-{attn_implementation}"
        subfolder = f"{benchmark_name}/{model.replace('/', '--')}"

        if not self.is_benchmark_supported(weights_config, attn_implementation):
            self.logger.info(f"Skipping benchmark {benchmark_name} with model {model} since it is not supported")
            return

        if self.is_benchmark_conducted(self.push_repo_id, subfolder):
            self.logger.info(f"Skipping benchmark {benchmark_name} with model {model} since it was already conducted")
            return

        benchmark_config = self.get_benchmark_config(model, attn_implementation, weights_config)
        benchmark_config.push_to_hub(repo_id=self.push_repo_id, subfolder=subfolder, private=True)
        self.execute_and_log_benchmark(benchmark_config, subfolder)

    @abstractmethod
    def get_benchmark_config(self, model: str, attn_implementation: str, weights_config: str) -> BenchmarkConfig:
        raise NotImplementedError("This method should be implemented in the child class")

    def execute_and_log_benchmark(self, benchmark_config: BenchmarkConfig, subfolder: str):
        try:
            self.logger.info(f"Running benchmark {benchmark_config.name} with model {benchmark_config.backend.model}")
            benchmark_report = Benchmark.launch(benchmark_config)
            benchmark_report.push_to_hub(repo_id=self.push_repo_id, subfolder=subfolder, private=True)
            benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
            benchmark.push_to_hub(repo_id=self.push_repo_id, subfolder=subfolder, private=True)
        except Exception:
            self.logger.error(f"Benchmark {benchmark_config.name} failed with model {benchmark_config.backend.model}")
            benchmark_report = BenchmarkReport.from_dict({"traceback": traceback.format_exc()})
            benchmark_report.push_to_hub(repo_id=self.push_repo_id, subfolder=subfolder, private=True)
            benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
            benchmark.push_to_hub(repo_id=self.push_repo_id, subfolder=subfolder, private=True)
