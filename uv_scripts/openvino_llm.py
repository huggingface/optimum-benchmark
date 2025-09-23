# /// script
# dependencies = [
#   "optimum-benchmark[openvino,ipex]@git+https://github.com/huggingface/optimum-benchmark.git@main",
#   "optimum-intel@git+https://github.com/huggingface/optimum-intel.git@main",
# ]
# ///

from argparse import ArgumentParser

from huggingface_hub import create_repo, upload_file

from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkReport,
    InferenceConfig,
    IPEXConfig,
    OpenVINOConfig,
    ProcessConfig,
    PyTorchConfig,
)
from optimum_benchmark.logging_utils import setup_logging
from optimum_benchmark.plot_utils import plot_decode_throughputs, plot_prefill_latencies

if __name__ == "__main__":
    setup_logging(level="INFO", prefix="MAIN-PROCESS")

    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="gpt2")
    parser.add_argument("--benchmark_repo_id", type=str, default=None)
    args = parser.parse_args()

    model_id = args.model_id
    benchmark_repo_id = args.benchmark_repo_id

    if benchmark_repo_id is not None:
        # not needed but useful to error early if benchmark_repo_id is not valid
        create_repo(benchmark_repo_id, repo_type="dataset", exist_ok=True)

    # Defining benchmark configurations
    launcher_config = ProcessConfig()
    scenario_config = InferenceConfig(
        latency=True,
        input_shapes={"batch_size": 1, "sequence_length": 16},
        generate_kwargs={"max_new_tokens": 16, "min_new_tokens": 16},
    )
    configs = {
        "ipex": IPEXConfig(device="cpu", model=model_id, no_weights=True),
        "openvino": OpenVINOConfig(device="cpu", model=model_id, no_weights=True),
        "pytorch-compile": PyTorchConfig(device="cpu", model=model_id, no_weights=True, torch_compile=True),
        "pytorch-compile-openvino": PyTorchConfig(
            device="cpu",
            model=model_id,
            no_weights=True,
            torch_compile=True,
            torch_compile_config={"backend": "openvino"},
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

        if benchmark_repo_id is not None:
            benchmark_report.push_to_hub(repo_id=benchmark_repo_id, filename=f"{config_name}_report.json")
            benchmark_config.push_to_hub(repo_id=benchmark_repo_id, filename=f"{config_name}_config.json")
        else:
            benchmark_report.save_json(f"{config_name}_report.json")
            benchmark_config.save_json(f"{config_name}_config.json")

    # Loading reports (from local files or from the hub if benchmark_repo_id is not None)
    reports = {}
    for config_name in configs.keys():
        if benchmark_repo_id is not None:
            reports[config_name] = BenchmarkReport.from_hub(
                repo_id=benchmark_repo_id, filename=f"{config_name}_report.json"
            )
        else:
            reports[config_name] = BenchmarkReport.from_json(f"{config_name}_report.json")

    # Plotting results (saved locally and uploaded to the hub if benchmark_repo_id is not None)
    fig1, ax1 = plot_prefill_latencies(reports)
    fig2, ax2 = plot_decode_throughputs(reports)
    fig1.savefig("prefill_latencies_boxplot.png")
    fig2.savefig("decode_throughput_barplot.png")

    if benchmark_repo_id is not None:
        upload_file(
            path_or_fileobj="prefill_latencies_boxplot.png",
            path_in_repo="plots/prefill_latencies_boxplot.png",
            repo_id=benchmark_repo_id,
            repo_type="dataset",
        )
        upload_file(
            path_or_fileobj="decode_throughput_barplot.png",
            path_in_repo="plots/decode_throughput_barplot.png",
            repo_id=benchmark_repo_id,
            repo_type="dataset",
        )
