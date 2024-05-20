<p align="center"><img src="https://raw.githubusercontent.com/huggingface/optimum-benchmark/main/logo.png" alt="Optimum-Benchmark Logo" width="350" style="max-width: 100%;" /></p>
<p align="center"><q>All benchmarks are wrong, some will cost you less than others.</q></p>
<h1 align="center">Optimum-Benchmark 🏋️</h1>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/optimum-benchmark)](https://pypi.org/project/optimum-benchmark/)
[![PyPI - Version](https://img.shields.io/pypi/v/optimum-benchmark)](https://pypi.org/project/optimum-benchmark/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/optimum-benchmark)](https://pypi.org/project/optimum-benchmark/)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/optimum-benchmark)](https://pypi.org/project/optimum-benchmark/)
[![PyPI - Format](https://img.shields.io/pypi/format/optimum-benchmark)](https://pypi.org/project/optimum-benchmark/)
[![PyPI - License](https://img.shields.io/pypi/l/optimum-benchmark)](https://pypi.org/project/optimum-benchmark/)

Optimum-Benchmark is a unified [multi-backend & multi-device](#backends--devices-) utility for benchmarking [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [PEFT](https://github.com/huggingface/peft), [TIMM](https://github.com/huggingface/pytorch-image-models) and [Optimum](https://github.com/huggingface/optimum) libraries, along with all their supported [optimizations & quantization schemes](#backends--devices-), for [inference & training](#scenarios-), in [distributed & non-distributed settings](#launchers-), in the most correct, efficient and scalable way possible.

*News* 📰

- 🥳 PyPI package is now available for installation: `pip install optimum-benchmark` 🎉 [check it out](https://pypi.org/project/optimum-benchmark/) !
- numactl support for Process and Torchrun launchers to control the NUMA nodes on which the benchmark runs 🧠
- 4 minimal docker images (`cpu`, `cuda`, `rocm`, `cuda-ort`) in [packages](https://github.com/huggingface/optimum-benchmark/pkgs/container/optimum-benchmark) for testing, benchmarking and reproducibility 🐳
- vLLM backend for benchmarking [vLLM](https://github.com/vllm-project/vllm)'s inference engine 🚀
- Hosting the codebase of the [LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard) 🥇
- Py-TXI backend for benchmarking [Py-TXI](https://github.com/IlyasMoutawwakil/py-txi/tree/main) 🚀
- Python API for running isolated and distributed benchmarks with Python scripts 🐍
- Simpler CLI interface for running benchmarks (runs and sweeps) using the Hydra 🧪

*Motivations* 🎯

- HuggingFace hardware partners wanting to know how their hardware performs compared to another hardware on the same models.
- HuggingFace ecosystem users wanting to know how their chosen model performs in terms of latency, throughput, memory usage, energy consumption, etc compared to another model.
- Benchmarking hardware & backend specific optimizations & quantization schemes that can be applied to models and improve their computational/memory/energy efficiency.

&#160;
> \[!Note\]
> Optimum-Benchmark is a work in progress and is not yet ready for production use, but we're working hard to make it so. Please keep an eye on the project and help us improve it and make it more useful for the community. We're looking forward to your feedback and contributions. 🚀
&#160;

## CI Status 🚦

Optimum-Benchmark is continuously and intensively tested on a variety of devices, backends, scenarios and launchers to ensure its stability with over 300 tests running on every PR (you can request more tests if you want to).

### API 📈

[![API_CPU](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_cpu.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_cpu.yaml)
[![API_CUDA](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_cuda.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_cuda.yaml)
[![API_MISC](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_misc.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_misc.yaml)
[![API_ROCM](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_rocm.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_rocm.yaml)

### CLI 📈

[![CLI_CPU_NEURAL_COMPRESSOR](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_neural_compressor.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_neural_compressor.yaml)
[![CLI_CPU_ONNXRUNTIME](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_onnxruntime.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_onnxruntime.yaml)
[![CLI_CPU_OPENVINO](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_openvino.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_openvino.yaml)
[![CLI_CPU_PYTORCH](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_pytorch.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_pytorch.yaml)
[![CLI_CPU_PY_TXI](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_py_txi.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_py_txi.yaml)
[![CLI_CUDA_ONNXRUNTIME](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_onnxruntime.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_onnxruntime.yaml)
[![CLI_CUDA_VLLM](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_vllm.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_vllm.yaml)
[![CLI_CUDA_PYTORCH_MULTI_GPU](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_pytorch_multi_gpu.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_pytorch_multi_gpu.yaml)
[![CLI_CUDA_PYTORCH_SINGLE_GPU](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_pytorch_single_gpu.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_pytorch_single_gpu.yaml)
[![CLI_CUDA_TENSORRT_LLM](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_tensorrt_llm.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_tensorrt_llm.yaml)
[![CLI_CUDA_TORCH_ORT_MULTI_GPU](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_torch_ort_multi_gpu.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_torch_ort_multi_gpu.yaml)
[![CLI_CUDA_TORCH_ORT_SINGLE_GPU](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_torch_ort_single_gpu.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_torch_ort_single_gpu.yaml)
[![CLI_MISC](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_misc.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_misc.yaml)
[![CLI_ROCM_PYTORCH_MULTI_GPU](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_rocm_pytorch_multi_gpu.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_rocm_pytorch_multi_gpu.yaml)
[![CLI_ROCM_PYTORCH_SINGLE_GPU](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_rocm_pytorch_single_gpu.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_rocm_pytorch_single_gpu.yaml)

## Quickstart 🚀

### Installation 📥

You can install the latest released version of `optimum-benchmark` on PyPI:

```bash
pip install optimum-benchmark
```

or you can install the latest version from the main branch on GitHub:

```bash
pip install git+https://github.com/huggingface/optimum-benchmark.git
```

or if you want to tinker with the code, you can clone the repository and install it in editable mode:

```bash
git clone https://github.com/huggingface/optimum-benchmark.git
cd optimum-benchmark
pip install -e .
```

<details>
    <summary>Advanced install options</summary>

Depending on the backends you want to use, you can install `optimum-benchmark` with the following extras:

- PyTorch (default): `pip install optimum-benchmark`
- OpenVINO: `pip install optimum-benchmark[openvino]`
- Torch-ORT: `pip install optimum-benchmark[torch-ort]`
- OnnxRuntime: `pip install optimum-benchmark[onnxruntime]`
- TensorRT-LLM: `pip install optimum-benchmark[tensorrt-llm]`
- OnnxRuntime-GPU: `pip install optimum-benchmark[onnxruntime-gpu]`
- Neural Compressor: `pip install optimum-benchmark[neural-compressor]`
- Py-TXI: `pip install optimum-benchmark[py-txi]`
- vLLM: `pip install optimum-benchmark[vllm]`

We also support the following extra extra dependencies:

- autoawq
- auto-gptq
- sentence-transformers
- bitsandbytes
- codecarbon
- flash-attn
- deepspeed
- diffusers
- timm
- peft

</details>

### Running benchmarks using the Python API 🧪

You can run benchmarks from the Python API, using the `Benchmark` class and its `launch` method. It takes a `BenchmarkConfig` object as input, runs the benchmark in an isolated process and returns a `BenchmarkReport` object containing the benchmark results.

Here's an example of how to run an isolated benchmark using the `pytorch` backend, `torchrun` launcher and `inference` scenario with latency and memory tracking enabled.

```python
from optimum_benchmark import Benchmark, BenchmarkConfig, TorchrunConfig, InferenceConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

setup_logging(level="INFO", handlers=["console"])

if __name__ == "__main__":
    launcher_config = TorchrunConfig(nproc_per_node=2)
    scenario_config = InferenceConfig(latency=True, memory=True)
    backend_config = PyTorchConfig(model="gpt2", device="cuda", device_ids="0,1", no_weights=True)
    benchmark_config = BenchmarkConfig(
        name="pytorch_gpt2",
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )
    benchmark_report = Benchmark.launch(benchmark_config)

    # log the benchmark in terminal
    benchmark_report.log() # or print(benchmark_report)

    # convert artifacts to a dictionary or dataframe
    benchmark_config.to_dict() # or benchmark_config.to_dataframe()

    # save artifacts to disk as json or csv files
    benchmark_report.save_csv("benchmark_report.csv") # or benchmark_report.save_json("benchmark_report.json")

    # push artifacts to the hub
    benchmark_config.push_to_hub("IlyasMoutawwakil/pytorch_gpt2") # or benchmark_config.push_to_hub("IlyasMoutawwakil/pytorch_gpt2")

    # or merge them into a single artifact
    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
    benchmark.save_json("benchmark.json") # or benchmark.save_csv("benchmark.csv")
    benchmark.push_to_hub("IlyasMoutawwakil/pytorch_gpt2")

    # load artifacts from the hub
    benchmark = Benchmark.from_hub("IlyasMoutawwakil/pytorch_gpt2") # or Benchmark.from_hub("IlyasMoutawwakil/pytorch_gpt2")

    # or load them from disk
    benchmark = Benchmark.load_json("benchmark.json") # or Benchmark.load_csv("benchmark_report.csv")
```

If you're on VSCode, you can hover over the configuration classes to see the available parameters and their descriptions. You can also see the available parameters in the [Features](#features-) section below.

### Running benchmarks using the Hydra CLI 🧪

You can also run a benchmark using the command line by specifying the configuration directory and the configuration name. Both arguments are mandatory for [`hydra`](https://hydra.cc/). `--config-dir` is the directory where the configuration files are stored and `--config-name` is the name of the configuration file without its `.yaml` extension.

```bash
optimum-benchmark --config-dir examples/ --config-name pytorch_bert
```

This will run the benchmark using the configuration in [`examples/pytorch_bert.yaml`](examples/pytorch_bert.yaml) and store the results in `runs/pytorch_bert`.

The resulting files are :

- `benchmark_config.json` which contains the configuration used for the benchmark, including the backend, launcher, scenario and the environment in which the benchmark was run.
- `benchmark_report.json` which contains a full report of the benchmark's results, like latency measurements, memory usage, energy consumption, etc.
- `benchmark.json` contains both the report and the configuration in a single file.
- `benchmark.log` contains the logs of the benchmark run.

<details>
<summary>Advanced CLI options</summary>

#### Configuration overrides 🎛️

It's easy to override the default behavior of a benchmark from the command line of an already existing configuration file. For example, to run the same benchmark on a different device, you can use the following command:

```bash
optimum-benchmark --config-dir examples/ --config-name pytorch_bert backend.model=gpt2 backend.device=cuda
```

#### Configuration sweeps 🧹

You can easily run configuration sweeps using the `--multirun` option. By default, configurations will be executed serially but other kinds of executions are supported with hydra's launcher plugins (e.g. `hydra/launcher=joblib`).

```bash
optimum-benchmark --config-dir examples --config-name pytorch_bert -m backend.device=cpu,cuda
```

### Configurations structure 📁

You can create custom and more complex configuration files following these [examples]([examples](https://github.com/IlyasMoutawwakil/optimum-benchmark-examples)). They are heavily commented to help you understand the structure of the configuration files.

</details>

## Features 🎨

`optimum-benchmark` allows you to run benchmarks with minimal configuration. A benchmark is defined by three main components:

- The launcher to use (e.g. `process`)
- The scenario to follow (e.g. `training`)
- The backend to run on (e.g. `onnxruntime`)

### Launchers 🚀

- [x] Process launcher (`launcher=process`); Launches the benchmark in an isolated process.
- [x] Torchrun launcher (`launcher=torchrun`); Launches the benchmark in multiples processes using `torch.distributed`.
- [x] Inline launcher (`launcher=inline`), not recommended for benchmarking, only for debugging purposes.

<details>
<summary>General Launcher features 🧰</summary>

- [x] Assert GPU devices (NVIDIA & AMD) isolation (`launcher.device_isolation=true`). This feature makes sure no other processes are running on the targeted GPU devices other than the benchmark. Espepecially useful when running benchmarks on shared resources.

</details>

### Scenarios 🏋

- [x] Training scenario (`scenario=training`) which benchmarks the model using the trainer class with a randomly generated dataset.
- [x] Inference scenario (`scenario=inference`) which benchmakrs the model's inference method (forward/call/generate) with randomly generated inputs.

<details>
<summary>Inference scenario features 🧰</summary>

- [x] Memory tracking (`scenario.memory=true`)
- [x] Energy and efficiency tracking (`scenario.energy=true`)
- [x] Latency and throughput tracking (`scenario.latency=true`)
- [x] Warm up runs before inference (`scenario.warmup_runs=20`)
- [x] Inputs shapes control (e.g. `scenario.input_shapes.sequence_length=128`)
- [x] Forward, Call and Generate kwargs (e.g. for an LLM `scenario.generate_kwargs.max_new_tokens=100`, for a diffusion model `scenario.call_kwargs.num_images_per_prompt=4`)

See [InferenceConfig](optimum_benchmark/scenarios/inference/config.py) for more information.

</details>

<details>
<summary>Training scenario features 🧰</summary>

- [x] Memory tracking (`scenario.memory=true`)
- [x] Energy and efficiency tracking (`scenario.energy=true`)
- [x] Latency and throughput tracking (`scenario.latency=true`)
- [x] Warm up steps before training (`scenario.warmup_steps=20`)
- [x] Dataset shapes control (e.g. `scenario.dataset_shapes.sequence_length=128`)
- [x] Training arguments control (e.g. `scenario.training_args.per_device_train_batch_size=4`)

See [TrainingConfig](optimum_benchmark/scenarios/training/config.py) for more information.

</details>

### Backends & Devices 📱

- [x] Pytorch backend for CPU (`backend=pytorch`, `backend.device=cpu`)
- [x] Pytorch backend for CUDA (`backend=pytorch`, `backend.device=cuda`, `backend.device_ids=0,1`)
- [ ] Pytorch backend for Habana Gaudi Processor (`backend=pytorch`, `backend.device=hpu`, `backend.device_ids=0,1`)
- [x] OnnxRuntime backend for CPUExecutionProvider (`backend=onnxruntime`, `backend.device=cpu`)
- [x] OnnxRuntime backend for CUDAExecutionProvider (`backend=onnxruntime`, `backend.device=cuda`)
- [x] OnnxRuntime backend for ROCMExecutionProvider (`backend=onnxruntime`, `backend.device=cuda`, `backend.provider=ROCMExecutionProvider`)
- [x] OnnxRuntime backend for TensorrtExecutionProvider (`backend=onnxruntime`, `backend.device=cuda`, `backend.provider=TensorrtExecutionProvider`)
- [x] Py-TXI backend for CPU and GPU (`backend=py-txi`, `backend.device=cpu` or `backend.device=cuda`)
- [x] Neural Compressor backend for CPU (`backend=neural-compressor`, `backend.device=cpu`)
- [x] TensorRT-LLM backend for CUDA (`backend=tensorrt-llm`, `backend.device=cuda`)
- [x] Torch-ORT backend for CUDA (`backend=torch-ort`, `backend.device=cuda`)
- [x] OpenVINO backend for CPU (`backend=openvino`, `backend.device=cpu`)
- [x] OpenVINO backend for GPU (`backend=openvino`, `backend.device=gpu`)
- [x] vLLM backend for CUDA (`backend=vllm`, `backend.device=cuda`)
- [x] vLLM backend for ROCM (`backend=vllm`, `backend.device=rocm`)
- [x] vLLM backend for CPU (`backend=vllm`, `backend.device=cpu`)

<details>
<summary>General backend features 🧰</summary>

- [x] Device selection (`backend.device=cuda`), can be `cpu`, `cuda`, `mps`, etc.
- [x] Device ids selection (`backend.device_ids=0,1`), can be a list of device ids to run the benchmark on multiple devices.
- [x] Model selection (`backend.model=gpt2`), can be a model id from the HuggingFace model hub or an **absolute path** to a model folder.
- [x] "No weights" feature, to benchmark models without downloading their weights, using randomly initialized weights (`backend.no_weights=true`)

</details>

<details>
<summary>Backend specific features 🧰</summary>

For more information on the features of each backend, you can check their respective configuration files:

- [VLLMConfig](optimum_benchmark/backends/vllm/config.py)
- [OVConfig](optimum_benchmark/backends/openvino/config.py)
- [PyTXIConfig](optimum_benchmark/backends/py_txi/config.py)
- [PyTorchConfig](optimum_benchmark/backends/pytorch/config.py)
- [ORTConfig](optimum_benchmark/backends/onnxruntime/config.py)
- [TorchORTConfig](optimum_benchmark/backends/torch_ort/config.py)
- [LLMSwarmConfig](optimum_benchmark/backends/llm_swarm/config.py)
- [TRTLLMConfig](optimum_benchmark/backends/tensorrt_llm/config.py)
- [INCConfig](optimum_benchmark/backends/neural_compressor/config.py)

</details>

## Contributing 🤝

Contributions are welcome! And we're happy to help you get started. Feel free to open an issue or a pull request.
Things that we'd like to see:

- More backends (Tensorflow, TFLite, Jax, etc).
- More tests (for optimizations and quantization schemes).
- More hardware support (Habana Gaudi Processor (HPU), Apple M series, etc).
- Task evaluators for the most common tasks (would be great for output regression).

To get started, you can check the [CONTRIBUTING.md](CONTRIBUTING.md) file.
