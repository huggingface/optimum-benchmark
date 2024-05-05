from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging(level="INFO")

    BENCHMARK_NAME = "pytorch_llama_awq"
    REPO_ID = f"IlyasMoutawwakil/{BENCHMARK_NAME}"

    launcher_config = ProcessConfig(
        device_isolation=True,
        device_isolation_action="warn",
    )
    scenario_config = InferenceConfig(
        memory=True,
        latency=True,
        input_shapes={"batch_size": 1, "sequence_length": 128},
        generate_kwargs={"max_new_tokens": 100, "min_new_tokens": 100},
    )
    backend_config = PyTorchConfig(
        device="cuda",
        device_ids="0",
        no_weights=True,
        model="TheBloke/Llama-2-70B-AWQ",
    )

    benchmark_config = BenchmarkConfig(
        name=BENCHMARK_NAME, launcher=launcher_config, scenario=scenario_config, backend=backend_config
    )
    # benchmark_config.push_to_hub(repo_id=REPO_ID)

    benchmark_report = Benchmark.launch(benchmark_config)
    # benchmark_report.push_to_hub(repo_id=REPO_ID)

    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
    # benchmark.push_to_hub(repo_id=REPO_ID)
