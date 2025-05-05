import os

from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

BENCHMARK_NAME = "cuda_pytorch_llama"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PUSH_REPO_ID = os.environ.get("PUSH_REPO_ID", None)

WEIGHTS_CONFIGS = {
    "float16": {
        "torch_dtype": "float16",
        "quantization_config": {},
    },
    "4bit-gptq-exllama-v2": {
        "torch_dtype": "float16",
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 4,
            "use_exllama ": True,
            "version": 2,
            "model_seqlen": 256,
        },
    },
    "torchao-int4wo-128": {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "torchao",
            "quant_type": "int4_weight_only",
            "group_size": 128,
        },
    },
}


def run_benchmark(weight_config: str):
    launcher_config = ProcessConfig(device_isolation=True, device_isolation_action="warn")
    backend_config = PyTorchConfig(
        model=MODEL,
        device="cuda",
        device_ids="0",
        no_weights=True,
        **WEIGHTS_CONFIGS[weight_config],
    )
    scenario_config = InferenceConfig(
        memory=True,
        latency=True,
        duration=10,
        iterations=10,
        warmup_runs=10,
        input_shapes={"batch_size": 1, "sequence_length": 64},
        generate_kwargs={"max_new_tokens": 32, "min_new_tokens": 32},
    )
    benchmark_config = BenchmarkConfig(
        name=BENCHMARK_NAME,
        launcher=launcher_config,
        scenario=scenario_config,
        backend=backend_config,
        print_report=True,
        log_report=True,
    )
    benchmark_report = Benchmark.launch(benchmark_config)

    return benchmark_config, benchmark_report


if __name__ == "__main__":
    level = os.environ.get("LOG_LEVEL", "INFO")
    to_file = os.environ.get("LOG_TO_FILE", "0") == "1"
    setup_logging(level=level, to_file=to_file, prefix="MAIN-PROCESS")

    for weight_config in WEIGHTS_CONFIGS:
        benchmark_config, benchmark_report = run_benchmark(weight_config)
        benchmark = Benchmark(config=benchmark_config, report=benchmark_report)

        if PUSH_REPO_ID is not None:
            benchmark.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=BENCHMARK_NAME, filename=f"{weight_config}.json")
