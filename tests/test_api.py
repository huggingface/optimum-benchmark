import gc
import os
import time
from importlib import reload
from logging import getLogger
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import torch

from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, ProcessConfig, PyTorchConfig, TrainingConfig
from optimum_benchmark.import_utils import get_git_revision_hash
from optimum_benchmark.system_utils import is_nvidia_system, is_rocm_system
from optimum_benchmark.trackers import LatencySessionTracker, MemoryTracker

LOGGER = getLogger("test-api")

os.environ["TRANSFORMERS_IS_CI"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PUSH_REPO_ID = os.environ.get("PUSH_REPO_ID", "optimum-benchmark/local")

LIBRARIES_TASKS_MODELS = [
    ("timm", "image-classification", "timm/tiny_vit_21m_224.in1k"),
    ("transformers", "fill-mask", "hf-internal-testing/tiny-random-BertModel"),
    ("transformers", "text-generation", "hf-internal-testing/tiny-random-LlamaForCausalLM"),
    ("diffusers", "text-to-image", "hf-internal-testing/tiny-stable-diffusion-torch"),
]

INPUT_SHAPES = {
    "batch_size": 2,  # for all tasks
    "sequence_length": 16,  # for text processing tasks
    "num_choices": 2,  # for multiple-choice task
}

DATASET_SHAPES = {
    "dataset_size": 2,  # for all tasks
    "sequence_length": 16,  # for text processing tasks
    "num_choices": 2,  # for multiple-choice task
}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("scenario", ["training", "inference"])
@pytest.mark.parametrize("library,task,model", LIBRARIES_TASKS_MODELS)
def test_api_launch(device, scenario, library, task, model):
    if scenario == "training" and library != "transformers":
        pytest.skip("Training is only supported with transformers library models")

    benchmark_name = f"{device}_{scenario}_{library}_{task}_{model}"

    if device == "cuda":
        device_isolation = True
        if is_rocm_system():
            device_isolation_action = "warn"
            device_ids = os.environ.get("ROCR_VISIBLE_DEVICES", "0")
        elif is_nvidia_system():
            device_isolation_action = "error"
            device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        else:
            raise RuntimeError("Using CUDA device on a machine that is neither NVIDIA nor ROCM.")
    else:
        device_isolation_action = None
        device_isolation = False
        device_ids = None

    launcher_config = ProcessConfig(
        device_isolation=device_isolation,
        device_isolation_action=device_isolation_action,
    )

    if scenario == "training":
        scenario_config = TrainingConfig(
            memory=True,
            latency=True,
            energy=not is_rocm_system(),
            warmup_steps=2,
            max_steps=5,
        )

    elif scenario == "inference":
        scenario_config = InferenceConfig(
            energy=not is_rocm_system(),
            latency=True,
            memory=True,
            duration=1,
            iterations=1,
            warmup_runs=1,
            input_shapes=INPUT_SHAPES,
            generate_kwargs={"max_new_tokens": 2, "min_new_tokens": 2},
            call_kwargs={"num_inference_steps": 2},
        )

    no_weights = False if library != "transformers" else True

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
        print_report=True,
        log_report=True,
    )
    benchmark_report = Benchmark.launch(benchmark_config)

    benchmark_report.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=benchmark_name)
    benchmark_config.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=benchmark_name)

    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
    benchmark.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=benchmark_name)


def test_api_push_to_hub_mixin():
    benchmark_name = "test_api_push_to_hub_mixin"

    backend_config = PyTorchConfig(model="google-bert/bert-base-uncased", device="cpu")
    launcher_config = ProcessConfig(device_isolation=False)
    scenario_config = InferenceConfig(
        duration=1,
        iterations=1,
        warmup_runs=1,
        memory=True,
        latency=True,
        input_shapes=INPUT_SHAPES,
    )
    benchmark_config = BenchmarkConfig(
        name=benchmark_name,
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
        print_report=True,
        log_report=True,
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["pytorch", "other"])
def test_api_latency_tracker(device, backend):
    tracker = LatencySessionTracker(device=device, backend=backend)

    # Warmup
    with tracker.session():
        while tracker.elapsed() < 10:
            with tracker.track():
                time.sleep(1)

    latency = tracker.get_latency()
    latency.log()

    # Elapsed
    with tracker.session():
        while tracker.elapsed() < 2:
            with tracker.track():
                time.sleep(1)

    latency = tracker.get_latency()
    latency.log()

    assert latency.mean < 1.1
    assert latency.mean > 0.9
    assert len(latency.values) == 2

    # Count
    with tracker.session():
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
    if device == "cuda" and backend == "other" and is_rocm_system():
        pytest.skip("Measuring memory usage is only supported for PyTorch backend on ROCm system for now")

    if torch.cuda.is_available():
        reload(torch.cuda)

    if device == "cuda":
        if is_rocm_system():
            device_ids = os.environ.get("ROCR_VISIBLE_DEVICES", "0")
        elif is_nvidia_system():
            device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    else:
        device_ids = None

    tracker = MemoryTracker(device=device, backend=backend, device_ids=device_ids)

    with tracker.track():
        time.sleep(1)
        pass

    initial_memory = tracker.get_max_memory()
    initial_memory.log()

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
            # namespace is not visible to pynvml/amdsmi, so we use global vram instead of process specific.
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
