# /// script
# dependencies = [
#   "optimum-benchmark[openvino]@git+https://github.com/huggingface/optimum-benchmark.git@main",
#   "optimum-intel@git+https://github.com/huggingface/optimum-intel.git@main",
#   "transformers==4.55.*",
#   "torchvision",
#   "num2words",
# ]
# ///

from argparse import ArgumentParser

from huggingface_hub import create_repo, upload_file

from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkReport,
    InferenceConfig,
    OpenVINOConfig,
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
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="The model to benchmark.",
    )
    parser.add_argument(
        "--benchmark_repo_id",
        type=str,
        default="optimum-benchmark/OpenVINO-VLM-Benchmark",
        help="The repository to store the benchmark results. Pass an empty string to disable pushing to the hub.",
    )
    args = parser.parse_args()

    model_id = args.model_id
    benchmark_repo_id = args.benchmark_repo_id

    if benchmark_repo_id:
        create_repo(benchmark_repo_id, repo_type="dataset", exist_ok=True)

    # Defining benchmark configurations
    launcher_config = ProcessConfig()
    scenario_config = InferenceConfig(
        memory=True,
        latency=True,
        generate_kwargs={"max_new_tokens": 16, "min_new_tokens": 16},
        input_shapes={"batch_size": 1, "sequence_length": 16, "num_images": 1},
    )
    configs = {
        "pytorch": PyTorchConfig(device="cpu", model=model_id, no_weights=True),
        "openvino": OpenVINOConfig(device="cpu", model=model_id, no_weights=True),
        "openvino-8bit-woq": OpenVINOConfig(
            device="cpu",
            model=model_id,
            no_weights=True,
            quantization_config={"bits": 8, "num_samples": 1, "weight_only": True},
        ),
        "openvino-8bit-static": OpenVINOConfig(
            device="cpu",
            model=model_id,
            no_weights=True,
            quantization_config={"bits": 8, "num_samples": 1, "weight_only": False, "dataset": "contextual"},
        ),
    }

    # Running benchmarks (saved locally or pushed to the hub if benchmark_repo_id is not None)
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
    fig, ax = plot_latencies(
        reports,
        target_name="prefill",
        title=f"{model_id} - Prefill Latencies (TTFT)",
        xlabel="Configurations",
        ylabel="Latency",
    )
    fig.savefig("prefill_latencies_boxplot.png")
    fig, ax = plot_throughputs(
        reports,
        target_name="decode",
        title=f"{model_id} - Decode Throughput (TPS)",
        xlabel="Configurations",
        ylabel="Throughput",
    )
    fig.savefig("decode_throughput_barplot.png")

    if benchmark_repo_id:
        upload_file(
            path_or_fileobj="prefill_latencies_boxplot.png",
            path_in_repo="prefill_latencies_boxplot.png",
            repo_id=benchmark_repo_id,
            repo_type="dataset",
        )
        upload_file(
            path_or_fileobj="decode_throughput_barplot.png",
            path_in_repo="decode_throughput_barplot.png",
            repo_id=benchmark_repo_id,
            repo_type="dataset",
        )
