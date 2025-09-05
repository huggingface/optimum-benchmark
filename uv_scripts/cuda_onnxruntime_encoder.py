# /// script
# dependencies = [
#   "optimum-benchmark[onnxruntime-gpu]",
# ]
# ///

from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    InferenceConfig,
    ONNXRuntimeConfig,
    ProcessConfig,
    PyTorchConfig,
)
from optimum_benchmark.logging_utils import setup_logging

MODEL = "bert-base-uncased"  # could be any encoder model / embedding model
setup_logging(level="INFO", prefix="MAIN-PROCESS")

if __name__ == "__main__":
    launcher_config = ProcessConfig()
    scenario_config = InferenceConfig(
        latency=True,
        input_shapes={"batch_size": 1, "sequence_length": 256},
    )

    backends = {
        "pytorch": PyTorchConfig(device="cuda", no_weights=True, model=MODEL),
        "pytorch-compile": PyTorchConfig(
            device="cuda",
            no_weights=True,
            model=MODEL,
            torch_compile=True,
            torch_compile_config={"backend": "inductor", "fullgraph": True, "mode": "max-autotune"},
        ),
        "pytorch-compile-cudagraphs": PyTorchConfig(
            device="cuda",
            no_weights=True,
            model=MODEL,
            torch_compile=True,
            torch_compile_config={"backend": "cudagraphs", "fullgraph": True},
        ),
        "onnxruntime": ONNXRuntimeConfig(device="cuda", no_weights=True, model=MODEL, use_io_binding=True),
    }

    results = {}
    for backend_name, backend_config in backends.items():
        benchmark_config = BenchmarkConfig(
            name=f"{backend_name}_bert",
            launcher=launcher_config,
            scenario=scenario_config,
            backend=backend_config,
        )
        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_report.save_json(f"{backend_name}_bert_benchmark_report.json")
        results[backend_name] = benchmark_report

    for backend_name, benchmark_report in results.items():
        print("-" * 80)
        print(f"Results for {backend_name}:")
        print("- Forward Metrics:")
        benchmark_report.forward.log()
