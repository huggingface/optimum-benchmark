from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    InferenceConfig,
    ProcessConfig,
    LlamaCppConfig,
)
from optimum_benchmark.logging_utils import setup_logging

setup_logging(level="INFO", prefix="MAIN-PROCESS")

if __name__ == "__main__":
    BENCHMARK_NAME = "llama_cpp_llama"

    launcher_config = ProcessConfig()
    backend_config = LlamaCppConfig(
        device="cuda",
        model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        task="text-generation",
        filename="tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    )
    scenario_config = InferenceConfig(
        latency=True,
        memory=True,
        energy=True,
        input_shapes={
            "batch_size": 1,
            "sequence_length": 256,
            "vocab_size": 32000,
        },
        generate_kwargs={
            "max_new_tokens": 100,
            "min_new_tokens": 100,
        },
    )

    # Combine all configurations into the benchmark configuration
    benchmark_config = BenchmarkConfig(
        name=BENCHMARK_NAME,
        launcher=launcher_config,
        backend=backend_config,
        scenario=scenario_config,
    )

    # Launch the benchmark with the specified configuration
    benchmark_report = Benchmark.launch(benchmark_config)

    # Optionally, create a Benchmark object with the config and report
    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)

    # save artifacts to disk as json or csv files
    benchmark_report.save_csv("benchmark_report.csv") # or benchmark_report.save_json("benchmark_report.json")

    # If needed, you can push the benchmark results to a repository (commented out)
    # REPO_ID = f"YourUsername/{BENCHMARK_NAME}"
    # benchmark_config.push_to_hub(repo_id=REPO_ID)
    # benchmark_report.push_to_hub(repo_id=REPO_ID)
    # benchmark.push_to_hub(repo_id=REPO_ID)
