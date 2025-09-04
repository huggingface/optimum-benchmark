# /// script
# dependencies = [
#   "optimum-benchmark[openvoino]",
# ]
# ///

from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, OpenVINOConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

MODEL = "gpt2"  # could be any decoder model / LLM
setup_logging(level="INFO", prefix="MAIN-PROCESS")

if __name__ == "__main__":
    launcher_config = ProcessConfig()
    scenario_config = InferenceConfig(
        latency=True,
        input_shapes={"batch_size": 1, "sequence_length": 16},
        generate_kwargs={"max_new_tokens": 16, "min_new_tokens": 16},
    )

    backends = {
        "openvino": OpenVINOConfig(device="cpu", no_weights=True, model=MODEL),
        "pytorch": PyTorchConfig(device="cpu", no_weights=True, model=MODEL),
        "pytorch-compile": PyTorchConfig(device="cpu", no_weights=True, model=MODEL, torch_compile=True),
        "pytorch-compile-openvino": PyTorchConfig(
            device="cpu", no_weights=True, model=MODEL, torch_compile=True, torch_compile_config={"backend": "openvino"}
        ),
    }

    results = {}
    for backend_name, backend_config in backends.items():
        benchmark_config = BenchmarkConfig(
            name=f"{backend_name}_gpt2",
            launcher=launcher_config,
            scenario=scenario_config,
            backend=backend_config,
        )
        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_report.save_json(f"{backend_name}_gpt2_benchmark_report.json")
        results[backend_name] = benchmark_report

    for backend_name, benchmark_report in results.items():
        print("-" * 80)
        print(f"Results for {backend_name}:")
        print("- Decode Metrics:")
        benchmark_report.decode.log()
