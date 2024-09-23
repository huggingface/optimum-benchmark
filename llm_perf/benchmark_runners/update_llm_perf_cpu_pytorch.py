from itertools import product
from typing import Any, Dict, List

from llm_perf.common.benchmark_runner import LLMPerfBenchmarkManager
from llm_perf.common.utils import CANONICAL_PRETRAINED_OPEN_LLM_LIST, GENERATE_KWARGS, INPUT_SHAPES
from optimum_benchmark import PyTorchConfig
from optimum_benchmark.benchmark.config import BenchmarkConfig
from optimum_benchmark.launchers.process.config import ProcessConfig
from optimum_benchmark.scenarios.inference.config import InferenceConfig


class CPUPyTorchBenchmarkRunner(LLMPerfBenchmarkManager):
    def __init__(self):
        super().__init__(backend="pytorch", device="cpu")

        self.attention_configs = self._get_attention_configs()
        assert self.subset is not None, "SUBSET environment variable must be set for benchmarking"
        self.weights_configs = self._get_weights_configs(self.subset)

    def get_list_of_benchmarks_to_run(self) -> List[Dict[str, Any]]:
        return [
            {"model": model, "attn_implementation": attn_impl, "weights_config": weights_cfg}
            for model, attn_impl, weights_cfg in product(
                CANONICAL_PRETRAINED_OPEN_LLM_LIST, self.attention_configs, self.weights_configs.keys()
            )
        ]

    def get_benchmark_name(self, model: str, **kwargs) -> str:
        weights_config = kwargs["weights_config"]
        attn_implementation = kwargs["attn_implementation"]
        return f"{model}-{weights_config}-{attn_implementation}"

    def get_benchmark_config(self, model: str, **kwargs) -> BenchmarkConfig:
        weights_config = kwargs["weights_config"]
        attn_implementation = kwargs["attn_implementation"]

        assert (
            weights_config in self.weights_configs
        ), f"your config does not contain {weights_config}, adjust your _get_weights_configs to fix this issue"

        torch_dtype = self.weights_configs[weights_config]["torch_dtype"]
        quant_scheme = self.weights_configs[weights_config]["quant_scheme"]
        quant_config = self.weights_configs[weights_config]["quant_config"]

        launcher_config = ProcessConfig()
        scenario_config = InferenceConfig(
            memory=True,
            energy=True,
            latency=True,
            duration=10,
            iterations=10,
            warmup_runs=10,
            input_shapes=INPUT_SHAPES,
            generate_kwargs=GENERATE_KWARGS,
        )
        backend_config = PyTorchConfig(
            model=model,
            device="cpu",
            no_weights=True,
            library="transformers",
            task="text-generation",
            torch_dtype=torch_dtype,
            quantization_scheme=quant_scheme,
            quantization_config=quant_config,
            attn_implementation=attn_implementation,
            model_kwargs={"trust_remote_code": True},
        )

        return BenchmarkConfig(
            name=f"{weights_config}-{attn_implementation}",
            scenario=scenario_config,
            launcher=launcher_config,
            backend=backend_config,
        )

    def _get_weights_configs(self, subset) -> Dict[str, Dict[str, Any]]:
        if subset == "unquantized":
            return {
                "float32": {"torch_dtype": "float32", "quant_scheme": None, "quant_config": {}},
                "float16": {"torch_dtype": "float16", "quant_scheme": None, "quant_config": {}},
                "bfloat16": {"torch_dtype": "bfloat16", "quant_scheme": None, "quant_config": {}},
            }
        else:
            raise ValueError(f"Unknown subset: {subset}")

    def _get_attention_configs(self) -> List[str]:
        return ["eager", "sdpa"]

if __name__ == "__main__":
    runner = CPUPyTorchBenchmarkRunner()
    runner.run_benchmarks()
