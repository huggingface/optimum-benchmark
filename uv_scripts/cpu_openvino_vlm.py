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
    backend_configs = {
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

    for config_name, backend_config in backend_configs.items():
        benchmark_config = BenchmarkConfig(
            name=f"{config_name}",
            launcher=launcher_config,
            scenario=scenario_config,
            backend=backend_config,
        )
        report = Benchmark.launch(benchmark_config)
        report.save_json(f"{config_name}_report.json")
        # report.push_to_hub(repo_id="IlyasMoutawwakil/vlm_benchmark", filename=f"{config_name}_report")

    backend_reports = {}
    for config_name in backend_configs.keys():
        backend_reports[config_name] = BenchmarkReport.from_json(f"{config_name}_report.json")
        # backend_reports[config_name] = BenchmarkReport.from_hub(
        #     repo_id="IlyasMoutawwakil/vlm_benchmark", filename=f"{config_name}_report"
        # )

    _, ax = plt.subplots()
    ax.boxplot(
        [backend_reports[config_name].prefill.latency.values for config_name in backend_reports.keys()],
        tick_labels=backend_reports.keys(),
        showfliers=False,
    )
    plt.xticks(rotation=10)
    ax.set_ylabel("Latency (s)")
    ax.set_xlabel("Configurations")
    ax.set_title("Prefill Latencies")
    plt.savefig("prefill_latencies_boxplot.png")

    _, ax = plt.subplots()
    ax.boxplot(
        [backend_reports[config_name].per_token.latency.values for config_name in backend_reports.keys()],
        tick_labels=backend_reports.keys(),
        showfliers=False,
    )
    plt.xticks(rotation=10)
    ax.set_ylabel("Latency (s)")
    ax.set_xlabel("Configurations")
    ax.set_title("Per-token Latencies")
    plt.savefig("per_token_latencies_boxplot.png")

    _, ax = plt.subplots()
    ax.bar(
        list(backend_reports.keys()),
        [backend_reports[config_name].generate.memory.max_ram for config_name in backend_reports.keys()],
        color=["C0", "C1", "C2", "C3", "C4", "C5"],
    )
    plt.xticks(rotation=10)
    ax.set_title("Max RAM")
    ax.set_ylabel("RAM (MB)")
    ax.set_xlabel("Configurations")
    plt.savefig("max_ram_barplot.png")
