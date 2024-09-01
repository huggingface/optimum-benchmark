import gc
import os
import time
from importlib import reload
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import torch

from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig, TrainingConfig
from optimum_benchmark.backends.diffusers_utils import (
    extract_diffusers_shapes_from_model,
    get_diffusers_pretrained_config,
)
from optimum_benchmark.backends.timm_utils import extract_timm_shapes_from_config, get_timm_pretrained_config
from optimum_benchmark.backends.transformers_utils import (
    extract_transformers_shapes_from_artifacts,
    get_transformers_pretrained_config,
    get_transformers_pretrained_processor,
)
from optimum_benchmark.generators.dataset_generator import DatasetGenerator
from optimum_benchmark.generators.input_generator import InputGenerator
from optimum_benchmark.import_utils import get_git_revision_hash
from optimum_benchmark.scenarios.inference.config import INPUT_SHAPES
from optimum_benchmark.scenarios.training.config import DATASET_SHAPES
from optimum_benchmark.trackers import LatencyTracker, MemoryTracker

PUSH_REPO_ID = os.environ.get("PUSH_REPO_ID", "optimum-benchmark/local")

LIBRARIES_TASKS_MODELS = [
    ("timm", "image-classification", "timm/resnet50.a1_in1k"),
    ("transformers", "text-generation", "openai-community/gpt2"),
    ("transformers", "fill-mask", "google-bert/bert-base-uncased"),
    ("transformers", "multiple-choice", "FacebookAI/roberta-base"),
    ("transformers", "text-classification", "FacebookAI/roberta-base"),
    ("transformers", "token-classification", "microsoft/deberta-v3-base"),
    ("transformers", "image-classification", "google/vit-base-patch16-224"),
    ("diffusers", "text-to-image", "CompVis/stable-diffusion-v1-4"),
]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("scenario", ["training", "inference"])
@pytest.mark.parametrize("library,task,model", LIBRARIES_TASKS_MODELS)
def test_api_launch(device, scenario, library, task, model):
    benchmark_name = f"{device}_{scenario}_{library}_{task}_{model}"

    if device == "cuda":
        if torch.version.hip is not None:
            device_ids = os.environ.get("HIP_VISIBLE_DEVICES", "0")
        else:
            device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    else:
        device_ids = None

    device_isolation = device == "cuda"
    no_weights = False if library != "transformers" else True

    launcher_config = ProcessConfig(device_isolation=device_isolation, device_isolation_action="error")

    if scenario == "training":
        if library == "transformers":
            scenario_config = TrainingConfig(memory=True, latency=True, warmup_steps=2, max_steps=5)
        else:
            pytest.skip("Training scenario is only available for Transformers library")

    elif scenario == "inference":
        scenario_config = InferenceConfig(
            energy=torch.version.hip is None,
            latency=True,
            memory=True,
            duration=1,
            iterations=1,
            warmup_runs=1,
            input_shapes={"batch_size": 1, "sequence_length": 2},
            generate_kwargs={"max_new_tokens": 2, "min_new_tokens": 2},
            call_kwargs={"num_inference_steps": 2},
        )

    backend_config = PyTorchConfig(
        device=device,
        device_ids=device_ids,
        no_weights=no_weights,
        library=library,
        model=model,
        task=task,
    )
    benchmark_config = BenchmarkConfig(
        name=benchmark_name,
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = Benchmark.launch(benchmark_config)

    benchmark_report.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=benchmark_name)
    benchmark_config.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=benchmark_name)

    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
    benchmark.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=benchmark_name)


def test_api_push_to_hub_mixin():
    benchmark_name = "test_api_push_to_hub_mixin"

    scenario_config = InferenceConfig(memory=True, latency=True, duration=1, iterations=1, warmup_runs=1)
    backend_config = PyTorchConfig(model="google-bert/bert-base-uncased", device="cpu")
    launcher_config = ProcessConfig(device_isolation=False)
    benchmark_config = BenchmarkConfig(
        name=benchmark_name,
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = Benchmark.launch(benchmark_config)
    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)

    for artifact_name, artifact in [
        ("benchmark_config", benchmark_config),
        ("benchmark_report", benchmark_report),
        ("benchmark", benchmark),
    ]:
        with TemporaryDirectory() as tempdir:
            # dict/json api
            artifact.save_json(f"{tempdir}/{artifact_name}.json")
            assert os.path.exists(f"{tempdir}/{artifact_name}.json")
            from_json_artifact = artifact.__class__.from_json(f"{tempdir}/{artifact_name}.json")
            assert from_json_artifact.to_dict() == artifact.to_dict()

            # dataframe/csv api
            artifact.save_csv(f"{tempdir}/{artifact_name}.csv")
            assert os.path.exists(f"{tempdir}/{artifact_name}.csv")
            from_csv_artifact = artifact.__class__.from_csv(f"{tempdir}/{artifact_name}.csv")
            pd.testing.assert_frame_equal(from_csv_artifact.to_dataframe(), artifact.to_dataframe())

        # Hugging Face Hub API
        artifact.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=benchmark_name)
        from_hub_artifact = artifact.__class__.from_pretrained(repo_id=PUSH_REPO_ID, subfolder=benchmark_name)
        assert from_hub_artifact.to_dict() == artifact.to_dict()


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
    assert len(generated_dataset) == DATASET_SHAPES["dataset_size"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["pytorch", "other"])
def test_api_latency_tracker(device, backend):
    tracker = LatencyTracker(device=device, backend=backend)

    tracker.reset()
    while tracker.elapsed() < 2:
        with tracker.track():
            time.sleep(1)

    latency = tracker.get_latency()
    latency.log()

    assert latency.mean < 1.1
    assert latency.mean > 0.9
    assert len(latency.values) == 2

    tracker.reset()
    while tracker.count() < 2:
        with tracker.track():
            time.sleep(1)

    latency = tracker.get_latency()
    latency.log()

    assert latency.mean < 1.1
    assert latency.mean > 0.9
    assert len(latency.values) == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["pytorch", "other"])
def test_api_memory_tracker(device, backend):
    if torch.cuda.is_available():
        reload(torch.cuda)

    if device == "cuda":
        if torch.version.hip is not None:
            device_ids = os.environ.get("HIP_VISIBLE_DEVICES", "0")
        else:
            device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    else:
        device_ids = None

    tracker = MemoryTracker(device=device, backend=backend, device_ids=device_ids)

    tracker.reset()
    with tracker.track():
        time.sleep(1)
        pass

    initial_memory = tracker.get_max_memory()
    initial_memory.log()

    tracker.reset()
    with tracker.track():
        array = torch.randn((10000, 10000), dtype=torch.float64, device=device)
        expected_memory = array.nbytes / 1e6
        time.sleep(1)

    final_memory = tracker.get_max_memory()
    final_memory.log()

    if device == "cuda":
        if backend == "pytorch":
            measured_memory = final_memory.max_allocated - initial_memory.max_allocated
        else:
            # namespace is not visible to pynvml/amdsmi,
            # so we use global vram instead of process specific vram.
            measured_memory = final_memory.max_global_vram - initial_memory.max_global_vram
    else:
        measured_memory = final_memory.max_ram - initial_memory.max_ram

    assert measured_memory < expected_memory * 1.1
    assert measured_memory > expected_memory * 0.9

    del array

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


def test_git_revision_hash_detection():
    assert get_git_revision_hash("optimum_benchmark") is not None
