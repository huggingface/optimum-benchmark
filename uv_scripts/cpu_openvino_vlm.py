# /// script
# dependencies = [
#   "optimum-benchmark[openvino]==0.7",
#   "transformers==4.53",
#   "torchvision",
#   "num2words",
# ]
# ///

from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, OpenVINOConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

setup_logging(level="INFO", to_file=True, prefix="OPTIMUM-BENCHMARK")


if __name__ == "__main__":
    launcher_config = ProcessConfig()
    scenario_config = InferenceConfig(
        latency=True,
        input_shapes={"batch_size": 1, "sequence_length": 16, "num_images": 1},
        generate_kwargs={"max_new_tokens": 16, "min_new_tokens": 16},
    )

    model = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    backend_configs = {
        "pytorch": PyTorchConfig(device="cpu", model=model, no_weights=True),
        "openvino": OpenVINOConfig(device="cpu", model=model, no_weights=True),
        "openvino-8bit-woq": OpenVINOConfig(
            device="cpu",
            model=model,
            no_weights=True,
            quantization_config={"bits": 8, "weight_only": True, "num_samples": 1},
        ),
        "openvino-8bit-static": OpenVINOConfig(
            device="cpu",
            model=model,
            no_weights=True,
            quantization_config={"n_bits": 8, "weight_only": False, "num_samples": 1},
        ),
    }

    results = {}
    for config_name, backend_config in backend_configs.items():
        benchmark_config = BenchmarkConfig(
            name=f"{config_name}",
            launcher=launcher_config,
            scenario=scenario_config,
            backend=backend_config,
        )
        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_report.save_json(f"{config_name}_vlm_benchmark_report.json")
        results[config_name] = benchmark_report

    for config_name, benchmark_report in results.items():
        print("-" * 80)
        print(f"Results for {config_name}:")
        print("- Prefill Metrics:")  # prefill = the processing of the input (text + image) to produce the first token
        benchmark_report.prefill.log()
        print("- Decode Metrics:")  # decode = the processing of subsequent tokens
        benchmark_report.decode.log()
