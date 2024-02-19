from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.launchers.torchrun.config import TorchrunConfig
from optimum_benchmark.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging(level="INFO")
    launcher_config = TorchrunConfig(nproc_per_node=2)
    benchmark_config = InferenceConfig(latency=True, memory=True)
    backend_config = PyTorchConfig(model="gpt2", device="cuda", device_ids="0,1", no_weights=True)
    experiment_config = ExperimentConfig(
        experiment_name="api-launch",
        benchmark=benchmark_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = launch(experiment_config)
    experiment_config.push_to_hub("IlyasMoutawwakil/benchmarks")
    benchmark_report.push_to_hub("IlyasMoutawwakil/benchmarks")
