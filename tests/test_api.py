import gc
import os
import time
from tempfile import TemporaryDirectory

import pandas as pd
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
from optimum_benchmark.benchmarks.report import BenchmarkReport
from optimum_benchmark.benchmarks.training.config import DATASET_SHAPES, TrainingConfig
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.generators.dataset_generator import DatasetGenerator
from optimum_benchmark.generators.input_generator import InputGenerator
from optimum_benchmark.import_utils import get_git_revision_hash
from optimum_benchmark.launchers.process.config import ProcessConfig
from optimum_benchmark.system_utils import get_gpu_device_ids
from optimum_benchmark.trackers.latency import LatencyTracker
from optimum_benchmark.trackers.memory import MemoryTracker

PUSH_REPO_ID = os.environ.get("PUSH_REPO_ID", "optimum-benchmark/misc")

LIBRARIES_TASKS_MODELS = [
    ("timm", "image-classification", "timm/resnet50.a1_in1k"),
    ("transformers", "text-generation", "openai-community/gpt2"),
    ("transformers", "fill-mask", "google-bert/bert-base-uncased"),
    ("transformers", "multiple-choice", "FacebookAI/roberta-base"),
    ("transformers", "text-classification", "FacebookAI/roberta-base"),
    ("transformers", "token-classification", "microsoft/deberta-v3-base"),
    ("transformers", "image-classification", "google/vit-base-patch16-224"),
    ("diffusers", "stable-diffusion", "CompVis/stable-diffusion-v1-4"),
]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("benchmark", ["training", "inference"])
@pytest.mark.parametrize("library,task,model", LIBRARIES_TASKS_MODELS)
def test_api_launch(device, benchmark, library, task, model):
    no_weights = False if library != "transformers" else True
    device_ids = get_gpu_device_ids() if device == "cuda" else None
    launcher_config = ProcessConfig(
        device_isolation=True if device == "cuda" else False,
        device_isolation_action="kill",
    )

    if benchmark == "training":
        if library == "transformers":
            benchmark_config = TrainingConfig(memory=True, latency=True, warmup_steps=2, max_steps=5)
        else:
            return  # skip training for non-transformers models

    elif benchmark == "inference":
        benchmark_config = InferenceConfig(
            memory=True,
            latency=True,
            duration=1,
            iterations=1,
            warmup_runs=1,
            input_shapes={"batch_size": 1, "sequence_length": 2},
            generate_kwargs={"max_new_tokens": 2, "min_new_tokens": 2},
            call_kwargs={"num_inference_steps": 2},
            energy=torch.version.hip is None,
        )

    backend_config = PyTorchConfig(
        device=device,
        device_ids=device_ids,
        no_weights=no_weights,
        library=library,
        model=model,
        task=task,
    )

    experiment_name = f"{device}_{benchmark}_{library}_{task}_{model}"
    experiment_config = ExperimentConfig(
        experiment_name=experiment_name,
        benchmark=benchmark_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = launch(experiment_config)

    experiment_config.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=experiment_name)
    benchmark_report.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=experiment_name)


def test_api_push_to_hub_mixin():
    experiment_name = "test_api_push_to_hub_mixin"

    launcher_config = ProcessConfig(device_isolation=False)
    backend_config = PyTorchConfig(model="google-bert/bert-base-uncased", device="cpu")
    benchmark_config = InferenceConfig(memory=True, latency=True, duration=1, iterations=1, warmup_runs=1)

    experiment_config = ExperimentConfig(
        experiment_name=experiment_name,
        benchmark=benchmark_config,
        launcher=launcher_config,
        backend=backend_config,
    )

    benchmark_report = launch(experiment_config)

    with TemporaryDirectory() as tempdir:
        # dict/json api
        experiment_config.save_json(f"{tempdir}/experiment_config.json")
        assert os.path.exists(f"{tempdir}/experiment_config.json")
        from_json_experiment_config: ExperimentConfig = ExperimentConfig.from_json(f"{tempdir}/experiment_config.json")
        assert from_json_experiment_config.to_dict() == experiment_config.to_dict()

        # dataframe/csv api
        experiment_config.save_csv(f"{tempdir}/experiment_config.csv")
        assert os.path.exists(f"{tempdir}/experiment_config.csv")
        from_csv_experiment_config: ExperimentConfig = ExperimentConfig.from_csv(f"{tempdir}/experiment_config.csv")
        pd.testing.assert_frame_equal(from_csv_experiment_config.to_dataframe(), experiment_config.to_dataframe())

    # Hugging Face Hub API
    experiment_config.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=experiment_name)
    from_hub_experiment_config: ExperimentConfig = ExperimentConfig.from_pretrained(
        repo_id=PUSH_REPO_ID, subfolder=experiment_name
    )
    assert from_hub_experiment_config.to_dict() == experiment_config.to_dict()

    with TemporaryDirectory() as tempdir:
        # dict/json api
        benchmark_report.save_json(f"{tempdir}/benchmark_report.json")
        assert os.path.exists(f"{tempdir}/benchmark_report.json")
        from_json_benchmark_report: BenchmarkReport = BenchmarkReport.from_json(f"{tempdir}/benchmark_report.json")
        assert from_json_benchmark_report.to_dict() == benchmark_report.to_dict()

        # dataframe/csv api
        benchmark_report.save_csv(f"{tempdir}/benchmark_report.csv")
        assert os.path.exists(f"{tempdir}/benchmark_report.csv")
        from_csv_benchmark_report: BenchmarkReport = BenchmarkReport.from_csv(f"{tempdir}/benchmark_report.csv")
        pd.testing.assert_frame_equal(from_csv_benchmark_report.to_dataframe(), benchmark_report.to_dataframe())

    # Hugging Face Hub API
    benchmark_report.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=experiment_name)
    from_hub_benchmark_report: BenchmarkReport = BenchmarkReport.from_pretrained(
        repo_id=PUSH_REPO_ID, subfolder=experiment_name
    )
    assert from_hub_benchmark_report.to_dict() == benchmark_report.to_dict()


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
    device_ids = get_gpu_device_ids() if device == "cuda" else None
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

    final_memory = tracker.get_max_memory()
    final_memory.log()

    if device == "cuda":
        if backend == "pytorch":
            measured_memory = final_memory.max_allocated - initial_memory.max_allocated
        else:
            # because user namespace is not visible to pynvml/amdsmi, we use global vram
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
