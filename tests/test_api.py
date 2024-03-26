import gc
import time
from tempfile import TemporaryDirectory

import pytest
import torch

from optimum_benchmark.backends.diffusers_utils import (
    extract_diffusers_shapes_from_model,
    get_diffusers_pretrained_config,
)
from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.backends.timm_utils import extract_timm_shapes_from_config, get_timm_pretrained_config
from optimum_benchmark.backends.transformers_utils import (
    extract_transformers_shapes_from_artifacts,
    get_transformers_pretrained_config,
    get_transformers_pretrained_processor,
)
from optimum_benchmark.benchmarks.inference.config import INPUT_SHAPES, InferenceConfig
from optimum_benchmark.benchmarks.training.config import DATASET_SHAPES, TrainingConfig
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.generators.dataset_generator import DatasetGenerator
from optimum_benchmark.generators.input_generator import InputGenerator
from optimum_benchmark.import_utils import get_git_revision_hash
from optimum_benchmark.launchers.inline.config import InlineConfig
from optimum_benchmark.launchers.process.config import ProcessConfig
from optimum_benchmark.trackers.latency import LatencyTracker
from optimum_benchmark.trackers.memory import MemoryTracker

LIBRARIES_TASKS_MODELS = [
    ("transformers", "fill-mask", "bert-base-uncased"),
    ("transformers", "text-generation", "openai-community/gpt2"),
    ("transformers", "text2text-generation", "google-t5/t5-small"),
    ("transformers", "multiple-choice", "FacebookAI/roberta-base"),
    ("transformers", "feature-extraction", "distilbert-base-uncased"),
    ("transformers", "text-classification", "FacebookAI/roberta-base"),
    ("transformers", "token-classification", "microsoft/deberta-v3-base"),
    ("transformers", "image-classification", "google/vit-base-patch16-224"),
    ("transformers", "semantic-segmentation", "google/vit-base-patch16-224"),
    ("diffusers", "stable-diffusion", "CompVis/stable-diffusion-v1-4"),
    ("timm", "image-classification", "timm/resnet50.a1_in1k"),
]
BENCHMARKS = ["training", "inference"]
LAUNCHERS = ["inline", "process"]
BACKENDS = ["pytorch", "other"]
DEVICES = ["cpu", "cuda"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_api_latency_tracker(device, backend):
    expected_latency = 1
    tracker = LatencyTracker(device=device, backend=backend)

    for _ in range(2):
        with tracker.track():
            time.sleep(1)

    latency = tracker.get_latency()
    latency.log()

    assert latency.mean < expected_latency * 1.1
    assert latency.mean > expected_latency * 0.9


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_api_memory_tracker(device, backend):
    tracker = MemoryTracker(device=device, backend=backend)

    tracker.reset()
    with tracker.track():
        time.sleep(1)
        pass

    # the process consumes memory that we can't control
    initial_memory = tracker.get_max_memory()
    initial_memory.log()

    tracker.reset()
    with tracker.track():
        time.sleep(1)
        array = torch.randn((10000, 10000), dtype=torch.float64, device=device)
        expected_memory = array.nbytes / 1e6
        time.sleep(1)

    final_memory = tracker.get_max_memory()
    final_memory.log()

    if device == "cuda":
        if backend == "pytorch":
            measured_memory = final_memory.max_allocated - initial_memory.max_allocated
        else:
            measured_memory = final_memory.max_vram - initial_memory.max_vram
            if torch.version.hip is not None:
                return  # skip vram measurement for ROCm
    else:
        measured_memory = final_memory.max_ram - initial_memory.max_ram

    assert measured_memory < expected_memory * 1.1
    assert measured_memory > expected_memory * 0.9

    del array

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("library,task,model", LIBRARIES_TASKS_MODELS)
def test_api_launch(device, launcher, benchmark, library, task, model):
    device_ids = "0" if device == "cuda" else None

    if benchmark == "training":
        benchmark_config = TrainingConfig(memory=True, latency=True)
    elif benchmark == "inference":
        benchmark_config = InferenceConfig(memory=True, latency=True)

    if launcher == "inline":
        launcher_config = ProcessConfig(device_isolation=False)
    elif launcher == "process":
        launcher_config = InlineConfig(device_isolation=False)

    backend_config = PyTorchConfig(
        device_ids=device_ids,
        no_weights=True,
        device=device,
        task=task,
        model=model,
        library=library,
    )
    experiment_config = ExperimentConfig(
        experiment_name=f"{library}_{task}_{model}_{device}",
        benchmark=benchmark_config,
        launcher=launcher_config,
        backend=backend_config,
    )

    benchmark_report = launch(experiment_config)

    with TemporaryDirectory() as tempdir:
        for name, artifact in {"config": experiment_config, "report": benchmark_report}.items():
            artifact.to_dict()
            artifact.to_flat_dict()
            artifact.to_dataframe()
            artifact.to_csv(f"{tempdir}/{name}.csv")
            artifact.to_json(f"{tempdir}/{name}.json")


@pytest.mark.parametrize("library,task,model", LIBRARIES_TASKS_MODELS)
def test_api_input_generator(library, task, model):
    if library == "transformers":
        model_config = get_transformers_pretrained_config(model)
        model_processor = get_transformers_pretrained_processor(model)
        model_shapes = extract_transformers_shapes_from_artifacts(model_config, model_processor)
    elif library == "timm":
        model_config = get_timm_pretrained_config(model)
        model_shapes = extract_timm_shapes_from_config(model_config)
    elif library == "diffusers":
        model_config = get_diffusers_pretrained_config(model)
        model_shapes = extract_diffusers_shapes_from_model(model)
    else:
        raise ValueError(f"Unknown library {library}")

    input_generator = InputGenerator(task=task, input_shapes=INPUT_SHAPES, model_shapes=model_shapes)
    generated_inputs = input_generator()

    assert len(generated_inputs) > 0, "No inputs were generated"

    for key in generated_inputs:
        assert len(generated_inputs[key]) == INPUT_SHAPES["batch_size"], "Incorrect batch size"


@pytest.mark.parametrize("library,task,model", LIBRARIES_TASKS_MODELS)
def test_api_dataset_generator(library, task, model):
    if library == "transformers":
        model_config = get_transformers_pretrained_config(model=model)
        model_shapes = extract_transformers_shapes_from_artifacts(config=model_config)
    elif library == "timm":
        model_config = get_timm_pretrained_config(model)
        model_shapes = extract_timm_shapes_from_config(config=model_config)
    elif library == "diffusers":
        model_config = get_diffusers_pretrained_config(model)
        model_shapes = extract_diffusers_shapes_from_model(model)
    else:
        raise ValueError(f"Unknown library {library}")

    generator = DatasetGenerator(task=task, dataset_shapes=DATASET_SHAPES, model_shapes=model_shapes)
    generated_dataset = generator()

    assert len(generated_dataset) > 0, "No dataset was generated"

    len(generated_dataset) == DATASET_SHAPES["dataset_size"]


def test_git_revision_hash_detection():
    assert get_git_revision_hash("optimum_benchmark") is not None
