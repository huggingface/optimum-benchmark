from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.launchers.process.config import ProcessConfig
from optimum_benchmark.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging(level="INFO")
    launcher_config = ProcessConfig(device_isolation=False)
    benchmark_config = InferenceConfig(
        memory=True,
        latency=True,
        input_shapes={"batch_size": 4, "sequence_length": 128},
        generate_kwargs={"max_new_tokens": 128, "min_new_tokens": 128},
    )
    backend_config = PyTorchConfig(
        model="TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
        device="cuda",
        device_ids="0",
        no_weights=True,
        quantization_scheme="awq",
        quantization_config={"version": "exllama"},
    )
    experiment_config = ExperimentConfig(
        experiment_name="awq-exllamav2",
        benchmark=benchmark_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = launch(experiment_config)
    experiment_config.push_to_hub("IlyasMoutawwakil/awq-benchmarks")
    benchmark_report.push_to_hub("IlyasMoutawwakil/awq-benchmarks")
