from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging(level="INFO")

    BENCHMARK_NAME = "pytorch_bert"
    REPO_ID = f"IlyasMoutawwakil/{BENCHMARK_NAME}"

    launcher_config = ProcessConfig(device_isolation=True, device_isolation_action="warn")
    backend_config = PyTorchConfig(device="cuda", device_ids="0", no_weights=True, model="bert-base-uncased")
    scenario_config = InferenceConfig(memory=True, latency=True, input_shapes={"batch_size": 1, "sequence_length": 128})

    benchmark_config = BenchmarkConfig(
        name=BENCHMARK_NAME, launcher=launcher_config, backend=backend_config, scenario=scenario_config
    )
    # benchmark_config.push_to_hub(repo_id=REPO_ID)

    benchmark_report = Benchmark.launch(benchmark_config)
    # benchmark_report.push_to_hub(repo_id=REPO_ID)

    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
    # benchmark.push_to_hub(repo_id=REPO_ID)
