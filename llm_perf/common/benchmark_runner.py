import os
import traceback
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Dict, List, Optional

from llm_perf.common.utils import (
    CANONICAL_PRETRAINED_OPEN_LLM_LIST,
    OPEN_LLM_LIST,
    PRETRAINED_OPEN_LLM_LIST,
)
from optimum_benchmark import Benchmark, BenchmarkConfig, BenchmarkReport
from optimum_benchmark.logging_utils import setup_logging


class LLMPerfBenchmarkManager(ABC):
    def __init__(self, backend: str, device: str, subset: Optional[str] = None, machine: Optional[str] = None):
        self.backend = backend
        self.device = device
        self.subset = subset or os.getenv("SUBSET", None)
        self.machine = machine or os.getenv("MACHINE", None)
        self.logger = getLogger("llm-perf-backend")

        if self.machine is None and self.subset is None:
            self.push_repo_id = f"optimum-benchmark/llm-perf-{self.backend}-{self.device}-debug"
            self.canonical_pretrained_open_llm_list = ["gpt2"]
            self.subset = "unquantized"
        elif self.machine is not None and self.subset is not None:
            self.push_repo_id = f"optimum-benchmark/llm-perf-{self.backend}-{self.device}-{self.subset}-{self.machine}"
        else:
            raise ValueError("Either both MACHINE and SUBSET should be set for benchmarking or neither for debugging")

        self.logger.info(f"len(OPEN_LLM_LIST): {len(OPEN_LLM_LIST)}")
        self.logger.info(f"len(PRETRAINED_OPEN_LLM_LIST): {len(PRETRAINED_OPEN_LLM_LIST)}")
        self.logger.info(f"len(CANONICAL_PRETRAINED_OPEN_LLM_LIST): {len(CANONICAL_PRETRAINED_OPEN_LLM_LIST)}")

    @abstractmethod
    def _get_weights_configs(self, subset: str) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError("This method should be implemented in the child class")

    @abstractmethod
    def _get_attention_configs(self) -> List[str]:
        raise NotImplementedError("This method should be implemented in the child class")

    def is_benchmark_supported(self, **kwargs) -> bool:
        """
        Can be overridden by child classes to exclude unsupported configurations
        """
        return True

    @abstractmethod
    def get_list_of_benchmarks_to_run(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("This method should be implemented in the child class")

    def run_benchmarks(self):
        os.environ["LOG_TO_FILE"] = "0"
        os.environ["LOG_LEVEL"] = "INFO"
        setup_logging(level="INFO", prefix="MAIN-PROCESS")

        benchmarks_to_run = self.get_list_of_benchmarks_to_run()

        self.logger.info(
            f"Running a total of {len(benchmarks_to_run)} benchmarks, "
            f"with {len(CANONICAL_PRETRAINED_OPEN_LLM_LIST)} models"
        )

        for benchmark_name in benchmarks_to_run:
            assert "model" in benchmark_name, "each benchmark should have a model"

            self.run_benchmark(**benchmark_name)

    def is_benchmark_conducted(self, push_repo_id, subfolder):
        try:
            report = BenchmarkReport.from_pretrained(repo_id=push_repo_id, subfolder=subfolder)
            if "traceback" in report.to_dict():
                return False
            else:
                return True
        except Exception:
            return False

    @abstractmethod
    def get_benchmark_name(self, model: str, **kwargs) -> str:
        raise NotImplementedError("This method should be implemented in the child class")

    def run_benchmark(self, **kwargs):
        model = kwargs["model"]

        benchmark_name = self.get_benchmark_name(model, **kwargs)
        subfolder = f"{benchmark_name}/{model.replace('/', '--')}"

        if not self.is_benchmark_supported(**kwargs):
            self.logger.info(f"Skipping benchmark {benchmark_name} with model {model} since it is not supported")
            return

        if self.is_benchmark_conducted(self.push_repo_id, subfolder):
            self.logger.info(f"Skipping benchmark {benchmark_name} with model {model} since it was already conducted")
            return

        benchmark_config = self.get_benchmark_config(model, **kwargs)
        benchmark_config.push_to_hub(repo_id=self.push_repo_id, subfolder=subfolder, private=True)
        self.execute_and_log_benchmark(benchmark_config, subfolder)

    @abstractmethod
    def get_benchmark_config(self, model: str, **kwargs) -> BenchmarkConfig:
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
