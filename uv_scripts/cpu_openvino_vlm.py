# /// script
# dependencies = [
#   "optimum-benchmark[openvino]@git+https://github.com/huggingface/optimum-benchmark.git@main",
#   "optimum-intel@git+https://github.com/huggingface/optimum-intel.git@main",
#   "transformers==4.55.*",
#   "torchvision",
#   "num2words",
# ]
# ///

import matplotlib.pyplot as plt
from huggingface_hub import upload_file

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

setup_logging(level="INFO", to_file=False, prefix="MAIN-PROCESS")

if __name__ == "__main__":
    launcher_config = ProcessConfig()
    scenario_config = InferenceConfig(
        memory=True,
        latency=True,
        generate_kwargs={"max_new_tokens": 16, "min_new_tokens": 16},
        input_shapes={"batch_size": 1, "sequence_length": 16, "num_images": 1},
    )

    model = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    configs = {
        "pytorch": PyTorchConfig(device="cpu", model=model, no_weights=True),
        "openvino": OpenVINOConfig(device="cpu", model=model, no_weights=True),
        "openvino-8bit-woq": OpenVINOConfig(
            device="cpu",
            model=model,
            no_weights=True,
            quantization_config={"bits": 8, "num_samples": 1, "weight_only": True},
        ),
        "openvino-8bit-static": OpenVINOConfig(
            device="cpu",
            model=model,
            no_weights=True,
            quantization_config={"bits": 8, "num_samples": 1, "dataset": "contextual"},
        ),
    }

    for config_name, backend_config in configs.items():
        benchmark_config = BenchmarkConfig(
            name=f"{config_name}",
            launcher=launcher_config,
            scenario=scenario_config,
            backend=backend_config,
        )
        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_report.push_to_hub(repo_id="IlyasMoutawwakil/vlm_benchmark", filename=f"{config_name}_report")
        benchmark_config.push_to_hub(repo_id="IlyasMoutawwakil/vlm_benchmark", filename=f"{config_name}_config")

    reports = {}
    for config_name in configs.keys():
        reports[config_name] = BenchmarkReport.from_hub(
            repo_id="IlyasMoutawwakil/vlm_benchmark", filename=f"{config_name}_report"
        )

    # Plotting results
    _, ax = plt.subplots()
    ax.boxplot(
        [reports[config_name].prefill.latency.values for config_name in reports.keys()],
        tick_labels=reports.keys(),
        showfliers=False,
    )
    plt.xticks(rotation=10)
    ax.set_ylabel("Latency (s)")
    ax.set_xlabel("Configurations")
    ax.set_title("Prefill Latencies")
    plt.savefig("prefill_latencies_boxplot.png")

    _, ax = plt.subplots()
    ax.boxplot(
        [reports[config_name].per_token.latency.values for config_name in reports.keys()],
        tick_labels=reports.keys(),
        showfliers=False,
    )
    plt.xticks(rotation=10)
    ax.set_ylabel("Latency (s)")
    ax.set_xlabel("Configurations")
    ax.set_title("Per-token Latencies")
    plt.savefig("per_token_latencies_boxplot.png")

    _, ax = plt.subplots()
    ax.bar(
        list(reports.keys()),
        [reports[config_name].generate.memory.max_ram for config_name in reports.keys()],
        color=["C0", "C1", "C2", "C3", "C4", "C5"],
    )
    plt.xticks(rotation=10)
    ax.set_title("Max RAM")
    ax.set_ylabel("RAM (MB)")
    ax.set_xlabel("Configurations")
    plt.savefig("max_ram_barplot.png")

    _, ax = plt.subplots()
    ax.bar(
        list(reports.keys()),
        [reports[config_name].decode.throughput.value for config_name in reports.keys()],
        color=["C0", "C1", "C2", "C3", "C4", "C5"],
    )
    plt.xticks(rotation=10)
    ax.set_xlabel("Configurations")
    ax.set_title("Decoding Throughput")
    ax.set_ylabel("Throughput (tokens/s)")
    plt.savefig("decode_throughput_barplot.png")

    # Uploading plots to hub
    upload_file(
        path_or_fileobj="prefill_latencies_boxplot.png",
        path_in_repo="prefill_latencies_boxplot.png",
        repo_id="IlyasMoutawwakil/vlm_benchmark",
        repo_type="dataset",
        token=True,
    )
    upload_file(
        path_or_fileobj="per_token_latencies_boxplot.png",
        path_in_repo="per_token_latencies_boxplot.png",
        repo_id="IlyasMoutawwakil/vlm_benchmark",
        repo_type="dataset",
        token=True,
    )
    upload_file(
        path_or_fileobj="max_ram_barplot.png",
        path_in_repo="max_ram_barplot.png",
        repo_id="IlyasMoutawwakil/vlm_benchmark",
        repo_type="dataset",
        token=True,
    )
    upload_file(
        path_or_fileobj="decode_throughput_barplot.png",
        path_in_repo="decode_throughput_barplot.png",
        repo_id="IlyasMoutawwakil/vlm_benchmark",
        repo_type="dataset",
        token=True,
    )
