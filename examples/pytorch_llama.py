import os

from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

BENCHMARK_NAME = "pytorch-llama"

WEIGHTS_CONFIGS = {
    "float16": {
        "torch_dtype": "float16",
        "quantization_scheme": None,
        "quantization_config": {},
    },
    # "4bit-awq-gemm": {
    #     "torch_dtype": "float16",
    #     "quantization_scheme": "awq",
    #     "quantization_config": {"bits": 4, "version": "gemm"},
    # },
    # "4bit-gptq-exllama-v2": {
    #     "torch_dtype": "float16",
    #     "quantization_scheme": "gptq",
    #     "quantization_config": {"bits": 4, "use_exllama ": True, "version": 2, "model_seqlen": 256},
    # },
}


def run_benchmark(weight_config: str):
    launcher_config = ProcessConfig(device_isolation=True, device_isolation_action="warn")
    backend_config = PyTorchConfig(
        device="cuda",
        device_ids="0",
        no_weights=True,
        model="gpt2",
        **WEIGHTS_CONFIGS[weight_config],
    )
    scenario_config = InferenceConfig(
        memory=True,
        latency=True,
        duration=10,
        iterations=10,
        warmup_runs=10,
        input_shapes={"batch_size": 1, "sequence_length": 128},
        generate_kwargs={"max_new_tokens": 32, "min_new_tokens": 32},
    )

    benchmark_config = BenchmarkConfig(
        name=BENCHMARK_NAME, launcher=launcher_config, scenario=scenario_config, backend=backend_config
    )
    benchmark_report = Benchmark.launch(benchmark_config)
    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)

    filename = f"{BENCHMARK_NAME}-{backend_config.version}-{weight_config}.json"
    benchmark.push_to_hub(repo_id="optimum-benchmark/pytorch-llama", filename=filename)
    benchmark.save_json(path=f"benchmarks/{filename}")


if __name__ == "__main__":
    level = os.environ.get("LOG_LEVEL", "INFO")
    to_file = os.environ.get("LOG_TO_FILE", "0") == "1"
    setup_logging(level=level, to_file=to_file, prefix="MAIN-PROCESS")

    for weight_config in WEIGHTS_CONFIGS:
        run_benchmark(weight_config)
