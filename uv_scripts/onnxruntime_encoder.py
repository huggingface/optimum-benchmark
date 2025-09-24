# /// script
# dependencies = [
#   "optimum-benchmark[onnxruntime-gpu]@git+https://github.com/huggingface/optimum-benchmark.git@main",
# ]
# ///

from argparse import ArgumentParser

from huggingface_hub import create_repo, upload_file

from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkReport,
    InferenceConfig,
    ONNXRuntimeConfig,
    ProcessConfig,
    PyTorchConfig,
)
from optimum_benchmark.logging_utils import setup_logging
from optimum_benchmark.plot_utils import plot_latencies, plot_throughputs

if __name__ == "__main__":
    setup_logging(level="INFO", prefix="MAIN-PROCESS")

    parser = ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="bert-base-uncased",
        help="The model to benchmark.",
    )
    parser.add_argument(
        "--benchmark_repo_id",
        type=str,
        default="optimum-benchmark/OnnxRuntime-Encoder-Benchmark",
        help="The repository to store the benchmark results. Pass an empty to disable pushing to the hub.",
    )
    args = parser.parse_args()

    model_id = args.model_id
    benchmark_repo_id = args.benchmark_repo_id

    if benchmark_repo_id:
        create_repo(benchmark_repo_id, repo_type="dataset", exist_ok=True)

    launcher_config = ProcessConfig()
    scenario_config = InferenceConfig(
        latency=True,
        input_shapes={"batch_size": 1, "sequence_length": 256},
    )

    configs = {
        "pytorch": PyTorchConfig(device="cuda", no_weights=True, model=model_id),
        "pytorch-compile": PyTorchConfig(
            device="cuda",
            model=model_id,
            no_weights=True,
            torch_compile=True,
            torch_compile_config={"backend": "inductor", "fullgraph": True},
        ),
        "pytorch-compile-cudagraphs": PyTorchConfig(
            device="cuda",
            model=model_id,
            no_weights=True,
            torch_compile=True,
            torch_compile_config={"backend": "cudagraphs", "fullgraph": True},
        ),
        "onnxruntime": ONNXRuntimeConfig(
            device="cuda",
            model=model_id,
            no_weights=True,
            use_io_binding=True,
        ),
        "onnxruntime-o4": ONNXRuntimeConfig(
            device="cuda",
            model=model_id,
            no_weights=True,
            use_io_binding=True,
            auto_optimization="O4",
        ),
    }

    # Running benchmarks (saved locally and pushed to the hub if benchmark_repo_id is not None)
    for config_name, backend_config in configs.items():
        benchmark_config = BenchmarkConfig(
            name=f"{config_name}",
            launcher=launcher_config,
            scenario=scenario_config,
            backend=backend_config,
        )
        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark = Benchmark(config=benchmark_config, report=benchmark_report)

        benchmark_report.save_json(f"{config_name}_report.json")
        benchmark_config.save_json(f"{config_name}_config.json")
        benchmark.save_json(f"{config_name}_benchmark.json")
        if benchmark_repo_id:
            benchmark_report.push_to_hub(repo_id=benchmark_repo_id, filename=f"{config_name}_report.json")
            benchmark_config.push_to_hub(repo_id=benchmark_repo_id, filename=f"{config_name}_config.json")
            benchmark.push_to_hub(repo_id=benchmark_repo_id, filename=f"{config_name}_benchmark.json")

    # Loading reports (from local files or from the hub if benchmark_repo_id is not None)
    reports = {}
    for config_name in configs.keys():
        if benchmark_repo_id:
            reports[config_name] = BenchmarkReport.from_hub(
                repo_id=benchmark_repo_id, filename=f"{config_name}_report.json"
            )
        else:
            reports[config_name] = BenchmarkReport.from_json(f"{config_name}_report.json")

    # Plotting results (saved locally and uploaded to the hub if benchmark_repo_id is not None)
    fig1, ax1 = plot_latencies(
        reports,
        target_name="forward",
        title=f"{model_id} - Forward Pass Latencies",
        xlabel="Configurations",
        ylabel="Latency",
    )
    fig1.savefig("forward_latencies_boxplot.png")
    fig2, ax2 = plot_throughputs(
        reports,
        target_name="forward",
        title=f"{model_id} - Forward Pass Throughput",
        xlabel="Configurations",
        ylabel="Throughput",
    )
    fig2.savefig("forward_throughput_barplot.png")

    if benchmark_repo_id:
        upload_file(
            path_or_fileobj="forward_latencies_boxplot.png",
            path_in_repo="plots/forward_latencies_boxplot.png",
            repo_id=benchmark_repo_id,
            repo_type="dataset",
        )
        upload_file(
            path_or_fileobj="forward_throughput_barplot.png",
            path_in_repo="plots/forward_throughput_barplot.png",
            repo_id=benchmark_repo_id,
            repo_type="dataset",
        )
