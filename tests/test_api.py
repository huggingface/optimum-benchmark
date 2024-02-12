from logging import getLogger
import time

import torch
import pytest

from optimum_benchmark.trackers.memory import MemoryTracker
from optimum_benchmark.trackers.latency import LatencyTracker
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.launchers.inline.config import InlineConfig
from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.launchers.process.config import ProcessConfig
from optimum_benchmark.launchers.torchrun.config import TorchrunConfig
from optimum_benchmark.benchmarks.inference.config import INPUT_SHAPES
from optimum_benchmark.benchmarks.training.config import DATASET_SHAPES
from optimum_benchmark.generators.input_generator import InputGenerator
from optimum_benchmark.benchmarks.training.config import TrainingConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig
from optimum_benchmark.generators.dataset_generator import DatasetGenerator
from optimum_benchmark.task_utils import TEXT_GENERATION_TASKS, IMAGE_DIFFUSION_TASKS
from optimum_benchmark.backends.timm_utils import extract_timm_shapes_from_config, get_timm_pretrained_config
from optimum_benchmark.backends.transformers_utils import (
    extract_transformers_shapes_from_artifacts,
    get_transformers_pretrained_config,
)


LOGGER = getLogger("test-api")

DEVICES_BACKENDS = [
    ("cpu", "none"),
    ("cuda", "pytorch"),
]
LIBRARIES_TASKS_MODELS = [
    ("transformers", "fill-mask", "bert-base-uncased"),
    ("timm", "image-classification", "timm/resnet50.a1_in1k"),
    ("transformers", "text-generation", "openai-community/gpt2"),
    ("transformers", "text2text-generation", "google-t5/t5-small"),
    ("transformers", "multiple-choice", "FacebookAI/roberta-base"),
    ("transformers", "feature-extraction", "distilbert-base-uncased"),
    ("transformers", "text-classification", "FacebookAI/roberta-base"),
    ("transformers", "token-classification", "microsoft/deberta-v3-base"),
    ("transformers", "image-classification", "google/vit-base-patch16-224"),
    ("transformers", "semantic-segmentation", "google/vit-base-patch16-224"),
]
BENCHMARK_CONFIGS = [
    InferenceConfig(latency=True, memory=True, energy=True),
    TrainingConfig(latency=True, memory=True, energy=True),
]
LAUNCHER_CONFIGS = [
    InlineConfig(device_isolation=False),
    ProcessConfig(device_isolation=False),
    TorchrunConfig(nproc_per_node=2, device_isolation=False),
]


@pytest.mark.parametrize("device,backend", DEVICES_BACKENDS)
def test_api_latency_tracker(device, backend):
    expected_latency = 1
    tracker = LatencyTracker(device=device, backend=backend)

    for _ in range(2):
        with tracker.track():
            time.sleep(1)

    latencies_list = tracker.get_latencies_list()

    assert len(latencies_list) == 2
    assert latencies_list[0] > expected_latency * 0.9
    assert latencies_list[0] < expected_latency * 1.1


@pytest.mark.parametrize("device,backend", DEVICES_BACKENDS)
def test_api_memory_tracker(device, backend):
    tracker = MemoryTracker(device=device, backend=backend)

    with tracker.track():
        pass

    # the process consumes memory that we can't control
    if backend == "pytorch":
        initial_process_memory = tracker.get_max_memory_allocated_mb()
    else:
        initial_process_memory = tracker.get_max_memory_used_mb()

    with tracker.track():
        array = torch.ones((10000, 10000), dtype=torch.float64, device=device)
        expected_memory = array.nbytes / 1e6  # around 800 MB

    if backend == "pytorch":
        final_process_memory = tracker.get_max_memory_allocated_mb()
    else:
        final_process_memory = tracker.get_max_memory_used_mb()

    measured_memory = final_process_memory - initial_process_memory

    assert measured_memory < expected_memory * 1.1
    assert measured_memory > expected_memory * 0.9


@pytest.mark.parametrize("library,task,model", LIBRARIES_TASKS_MODELS)
def test_api_input_generator(library, task, model):
    if library == "transformers":
        model_config = get_transformers_pretrained_config(model)
        model_shapes = extract_transformers_shapes_from_artifacts(model_config)
    elif library == "timm":
        model_config = get_timm_pretrained_config(model)
        model_shapes = extract_timm_shapes_from_config(model_config)
    else:
        raise ValueError(f"Unknown library {library}")

    generator = InputGenerator(
        task=task,
        input_shapes=INPUT_SHAPES,
        model_shapes=model_shapes,
    )

    if task in TEXT_GENERATION_TASKS:
        _ = generator(mode="forward")
        _ = generator(mode="generate")
    elif task in IMAGE_DIFFUSION_TASKS:
        _ = generator(mode="call")
    else:
        _ = generator(mode="forward")


@pytest.mark.parametrize("library,task,model", LIBRARIES_TASKS_MODELS)
def test_api_dataset_generator(library, task, model):
    if library == "transformers":
        model_config = get_transformers_pretrained_config(model=model)
        model_shapes = extract_transformers_shapes_from_artifacts(config=model_config)
    elif library == "timm":
        model_config = get_timm_pretrained_config(model)
        model_shapes = extract_timm_shapes_from_config(config=model_config)
    else:
        raise ValueError(f"Unknown library {library}")

    generator = DatasetGenerator(
        task=task,
        dataset_shapes=DATASET_SHAPES,
        model_shapes=model_shapes,
    )

    _ = generator()


@pytest.mark.parametrize("launcher_config", LAUNCHER_CONFIGS)
def test_api_launchers(launcher_config):
    backend_config = PyTorchConfig(model="gpt2", no_weights=True, device="cpu")
    benchmark_config = InferenceConfig(memory=True)
    experiment_config = ExperimentConfig(
        experiment_name="api-launch-experiment",
        benchmark=benchmark_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    _ = launch(experiment_config)

@pytest.mark.parametrize("benchmark_config", BENCHMARK_CONFIGS)
@pytest.mark.parametrize("launcher_config", LAUNCHER_CONFIGS)
@pytest.mark.parametrize("backend_config", [PyTorchConfig(model="gpt2", no_weights=True, device="cpu")])
def test_api_benchmarks(benchmark_config, launcher_config, backend_config):
    experiment_config = ExperimentConfig(
        experiment_name="api-benchmark-experiment",
        benchmark=benchmark_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    _ = launch(experiment_config)
