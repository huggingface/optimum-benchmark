import os

from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

BENCHMARK_NAME = "cuda_pytorch_bert"
MODEL = "google-bert/bert-base-uncased"
PUSH_REPO_ID = os.environ.get("PUSH_REPO_ID", None)


if __name__ == "__main__":
    level = os.environ.get("LOG_LEVEL", "INFO")
    to_file = os.environ.get("LOG_TO_FILE", "0") == "1"
    setup_logging(level=level, to_file=to_file, prefix="MAIN-PROCESS")

    launcher_config = ProcessConfig(device_isolation=True, device_isolation_action="warn")
    backend_config = PyTorchConfig(device="cuda", device_ids="0", no_weights=True, model=MODEL)
    scenario_config = InferenceConfig(memory=True, latency=True, input_shapes={"batch_size": 1, "sequence_length": 128})
    benchmark_config = BenchmarkConfig(
        name=BENCHMARK_NAME,
        launcher=launcher_config,
        scenario=scenario_config,
        backend=backend_config,
        print_report=True,
        log_report=True,
    )
    benchmark_report = Benchmark.launch(benchmark_config)
    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)

    if PUSH_REPO_ID is not None:
        benchmark_config.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=BENCHMARK_NAME)
        benchmark_report.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=BENCHMARK_NAME)
        benchmark.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=BENCHMARK_NAME)
