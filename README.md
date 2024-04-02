<p align="center"><img src="logo.png" alt="Optimum-Benchmark Logo" width="350" style="max-width: 100%;" /></p>
<p align="center"><q>All benchmarks are wrong, some will cost you less than others.</q></p>
<h1 align="center">Optimum-Benchmark 🏋️</h1>

Optimum-Benchmark is a unified [multi-backend & multi-device](#backends--devices-) utility for benchmarking [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [PEFT](https://github.com/huggingface/peft), [TIMM](https://github.com/huggingface/pytorch-image-models) and [Optimum](https://github.com/huggingface/optimum) libraries, along with all their supported [optimizations & quantization schemes](#backend-features-), for [inference & training](#benchmarks-), in [distributed & non-distributed settings](#backend-), in the most correct, efficient and scalable way possible.

*News* 📰

- PYPI release soon.
- Added a Python API to run benchmarks with isolation, distribution and tracking features supported by the library.

*Motivations* 🤔

- HuggingFace hardware partners wanting to know how their hardware performs compared to another hardware on the same models.
- HuggingFace ecosystem users wanting to know how their chosen model performs in terms of latency, throughput, memory usage, energy consumption, etc compared to another model.
- Experimenting with hardware & backend specific optimizations & quantization schemes that can be applied to models and improve their computational/memory/energy efficiency.
- [...]

&#160;

> \[!Note\]
> Optimum-Benchmark is a work in progress and is not yet ready for production use, but we're working hard to make it so. Please keep an eye on the project and help us improve it and make it more useful for the community.

&#160;

## CI Status 🚦

Optimum-Benchmark is continuously and intensively tested on a variety of devices, backends, benchmarks and launchers to ensure its stability with over 300 tests running on every PR (you can request more tests if you want to).

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

You can install `optimum-benchmark` using `pip`:

```bash
pip install optimum-benchmark@git+https://github.com/huggingface/optimum-benchmark.git
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
- Intel Neural Compressor: `pip install optimum-benchmark[neural-compressor]`
- Py-TXI: `pip install optimum-benchmark[py-txi]`

</details>

### Running backend benchmarks using the Python API 🧪

You can run benchmarks from the Python API, using the `launch` entrypoint. It takes an `ExperimentConfig` object as input and returns a `BenchmarkReport` object containing the benchmark results. The use of configuration files is optional, but recommended for utmost correctness and reproducibility of benchmarks.

Here's an example of how to run a benchmark using the `pytorch` backend, `torchrun` launcher and `inference` benchmark.

```python
from optimum_benchmark.experiment import launch, ExperimentConfig
from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.launchers.torchrun.config import TorchrunConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig

if __name__ == "__main__":
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

    # push artifacts to the hub
    experiment_config.push_to_hub("IlyasMoutawwakil/benchmarks")
    benchmark_report.push_to_hub("IlyasMoutawwakil/benchmarks")
```

If you're on VSCode, you can hover over the configuration classes to see the available parameters and their descriptions. Documentation will be available soon (help is welcome!).

### Running backend benchmarks using the Hydra CLI 🧪

You can also run a benchmark using the command line by specifying the configuration directory and the configuration name. Both arguments are mandatory for [`hydra`](https://hydra.cc/). `--config-dir` is the directory where the configuration files are stored and `--config-name` is the name of the configuration file without its `.yaml` extension.

```bash
optimum-benchmark --config-dir examples/ --config-name pytorch_bert
```

This will run the benchmark using the configuration in [`examples/pytorch_bert.yaml`](examples/pytorch_bert.yaml) and store the results in `runs/pytorch_bert`.

The result files are `benchmark_report.json`, the program's logs `cli.log` and the configuration that's been used `experiment_config.json`, including backend, launcher, benchmark and environment information.

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

`optimum-benchmark` allows you to run backend benchmarks with minimal configuration. A backend benchmark is defined by three main components:

- The launcher to use (e.g. `process`)
- The benchmark to run (e.g. `training`)
- The backend to run on (e.g. `onnxruntime`)

### Launchers 🚀

- [x] Isolated process launcher (`launcher=process`).
- [x] Distributed inference/training launcher (`launcher=torchrun`).
- [x] Inline launcher (`launcher=inline`), not recommended for benchmarking.

<details>
<summary>General Launcher features 🧰</summary>

- [x] Assert GPU devices (NVIDIA & AMD) isolation (`launcher.device_isolation=true`). This feature makes sure no other processes are running on the targeted GPU devices other than the benchmark. Espepecially useful when running benchmarks on shared resources.

</details>

### Benchmarks 🏋

- [x] Training benchmark (`benchmark=training`) which benchmarks the model using the trainer class with a randomly generated dataset.
- [x] Inference benchmark (`benchmark=inference`) which benchmakrs the model's inference method (forward/call/generate) with randomly generated inputs.

<details>
<summary>Inference benchmark features 🧰</summary>

- [x] Memory tracking (`benchmark.memory=true`)
- [x] Energy and efficiency tracking (`benchmark.energy=true`)
- [x] Latency and throughput tracking (`benchmark.latency=true`)
- [x] Warm up runs before inference (`benchmark.warmup_runs=20`)
- [x] Inputs shapes control (e.g. `benchmark.input_shapes.sequence_length=128`)
- [x] Forward, Call and Generate kwargs (e.g. for an LLM `benchmark.generate_kwargs.max_new_tokens=100`, for a diffusion model `benchmark.call_kwargs.num_images_per_prompt=4`)

</details>

<details>
<summary>Training benchmark features 🧰</summary>

- [x] Memory tracking (`benchmark.memory=true`)
- [x] Energy and efficiency tracking (`benchmark.energy=true`)
- [x] Latency and throughput tracking (`benchmark.latency=true`)
- [x] Warm up steps before training (`benchmark.warmup_steps=20`)
- [x] Dataset shapes control (e.g. `benchmark.dataset_shapes.sequence_length=128`)
- [x] Training arguments control (e.g. `benchmark.training_args.per_device_train_batch_size=4`)

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
- [x] Intel Neural Compressor backend for CPU (`backend=neural-compressor`, `backend.device=cpu`)
- [x] TensorRT-LLM backend for CUDA (`backend=tensorrt-llm`, `backend.device=cuda`)
- [x] Torch-ORT backend for CUDA (`backend=torch-ort`, `backend.device=cuda`)
- [x] OpenVINO backend for CPU (`backend=openvino`, `backend.device=cpu`)
- [x] OpenVINO backend for GPU (`backend=openvino`, `backend.device=gpu`)


<details>
<summary>General backend features 🧰</summary>

- [x] Model selection (`backend.model=gpt2`), can be a model id from the HuggingFace model hub or an absolute path to a model folder.
- [x] Device selection (`backend.device=cuda`), can be `cpu`, `cuda`, `mps`, etc.
- [ ] Device ids selection (`backend.device_ids=0,1`), can be a list of device ids to run the benchmark on multiple devices.
- [x] "No weights" feature, to benchmark models without downloading their weights (`backend.no_weights=true`)

</details>

<details>
<summary>Backend specific features 🧰</summary>

For more information on the features of each backend, you can check their respective configuration files:

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
