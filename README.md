<p align="center"><img src="logo.png" alt="Optimum-Benchmark Logo" width="350" style="max-width: 100%;" /></p>
<p align="center"><q>All benchmarks are wrong, some will cost you less than others.</q></p>
<h1 align="center">Optimum-Benchmark 🏋️</h1>

Optimum-Benchmark is a unified [multi-backend & multi-device](#backends--devices-) utility for benchmarking [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [PEFT](https://github.com/huggingface/peft), [TIMM](https://github.com/huggingface/pytorch-image-models) and [Optimum](https://github.com/huggingface/optimum) flavors, along with all their supported [optimizations & quantization schemes](#backend-features-), for [inference & training](#benchmark-features-%EF%B8%8F), in [distributed & non-distributed settings](#backend-features-), in the most correct, efficient and scalable way possible (you don't even need to download the weights).

*News* 📰
- PYPI release soon.
- Added a simple Python API to run benchmarks with all isolation and tracking features supported by the CLI.

*Motivations* 🤔
- HF hardware partners wanting to know how their hardware performs compared to another hardware on the same models.
- HF ecosystem users wanting to know how their chosen model performs in terms of latency, throughput, memory usage, energy consumption, etc compared to another model.
- Experimenting with hardware & backend specific optimizations & quantization schemes that can be applied to models and improve their computational/memory/energy efficiency.

*Notes* 📝
- If you were using `optimum-benchmark` before and want to keep using the old CLI only version, you can still do so by installing from this branch [`0.0.1`](https://github.com/huggingface/optimum-benchmark/tree/0.0.1).

## Current status 📈

### API
[![CPU](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_cpu.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_cpu.yaml)
[![CUDA](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_cuda.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_cuda.yaml)
[![ROCM](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_rocm.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_rocm.yaml)
[![MISC](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_misc.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_api_misc.yaml)

### CLI
[![CPU Pytorch Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_pytorch.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_pytorch.yaml)
[![CPU OnnxRuntime Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_onnxruntime.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_onnxruntime.yaml)
[![CPU Intel Neural Compressor Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_neural_compressor.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_neural_compressor.yaml)
[![CPU OpenVINO Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_openvino.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cpu_openvino.yaml)
[![CUDA Pytorch Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_pytorch.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_pytorch.yaml)
[![CUDA OnnxRuntime Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_onnxruntime.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_onnxruntime.yaml)
[![CUDA Torch-ORT Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_torch_ort.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_cuda_torch_ort.yaml)
[![TensorRT OnnxRuntime Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_tensorrt_onnxruntime.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_tensorrt_onnxruntime.yaml)
[![TensorRT-LLM Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_tensorrt_llm.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_tensorrt_llm.yaml)
[![ROCm Pytorch Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_rocm_pytorch.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_rocm_pytorch.yaml)
[![ROCm OnnxRuntime Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_rocm_onnxruntime.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_rocm_onnxruntime.yaml)
[![MISC Tests](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_misc.yaml/badge.svg)](https://github.com/huggingface/optimum-benchmark/actions/workflows/test_cli_misc.yaml)

## Quickstart 🚀

### Installation 📥

You can install `optimum-benchmark` using pip:

```bash
pip install optimum-benchmark
```

or by cloning the repository and installing it in editable mode:

```bash
git clone https://github.com/huggingface/optimum-benchmark.git
cd optimum-benchmark
pip install -e .
```

Depending on the backends you want to use, you might need to install some extra dependencies:

- Pytorch (default): `pip install optimum-benchmark`
- OpenVINO: `pip install optimum-benchmark[openvino]`
- Torch-ORT: `pip install optimum-benchmark[torch-ort]`
- OnnxRuntime: `pip install optimum-benchmark[onnxruntime]`
- TensorRT-LLM: `pip install optimum-benchmark[tensorrt-llm]`
- OnnxRuntime-GPU: `pip install optimum-benchmark[onnxruntime-gpu]`
- Intel Neural Compressor: `pip install optimum-benchmark[neural-compressor]`
- Text Generation Inference: `pip install optimum-benchmark[text-generation-inference]`

### Running benchmarks from Python API 🧪

You can run benchmarks from the Python API, using the `launch` function. Here's an example of how to run a benchmark using the `pytorch` backend, `torchrun` launcher and `inference` benchmark.

```python
from optimum_benchmark.logging_utils import setup_logging
from optimum_benchmark.experiment import launch, ExperimentConfig
from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.launchers.torchrun.config import TorchrunConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig

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
    experiment_config.push_to_hub("IlyasMoutawwakil/benchmarks") # pushes experiment_config.json to the hub
    benchmark_report.push_to_hub("IlyasMoutawwakil/benchmarks") # pushes benchmark_report.json to the hub
```

Yep, it's that simple! Check the supported backends, launchers and benchmarks matrix in the [features](#features-) section.

### Running benchmarks from CLI 🏃‍♂️

You can also run a benchmark using the command line by specifying the configuration directory and the configuration name. Both arguments are mandatory for [`hydra`](https://hydra.cc/). `--config-dir` is the directory where the configuration files are stored and `--config-name` is the name of the configuration file without its `.yaml` extension.

```bash
optimum-benchmark --config-dir examples/ --config-name pytorch_bert
```

This will run the benchmark using the configuration in [`examples/pytorch_bert.yaml`](examples/pytorch_bert.yaml) and store the results in `runs/pytorch_bert`.

The result files are `benchmark_report.json`, the program's logs `cli.log` and the configuration that's been used `experiment_config.json`, including backend, launcher, benchmark and environment configurations.

The directory for storing these results can be changed by setting `hydra.run.dir` (and/or `hydra.sweep.dir` in case of a multirun) in the command line or in the config file.

#### Configuration overrides 🎛️

It's easy to override the default behavior of a benchmark from the command line.

```bash
optimum-benchmark --config-dir examples/ --config-name pytorch_bert backend.model=gpt2 backend.device=cuda
```

#### Configuration multirun sweeps 🧹

You can easily run configuration sweeps using the `-m` or `--multirun` option. By default, configurations will be executed serially but other kinds of executions are supported with hydra's launcher plugins : `=submitit`, `hydra/launcher=rays`, etc.

```bash
optimum-benchmark --config-dir examples --config-name pytorch_bert -m backend.device=cpu,cuda
```

### Configurations structure 📁

You can create custom and more complex configuration files following these [examples]([examples](https://github.com/IlyasMoutawwakil/optimum-benchmark-examples)).

## Features 🎨

`optimum-benchmark` allows you to run benchmarks with minimal configuration. The only required parameters are:

- The launcher to use (e.g. `process`).
- The type of benchmark (e.g. `training`)
- The backend to run on (e.g. `onnxruntime`).
- The model name or path (e.g. `bert-base-uncased`)

Everything else is optional or inferred at runtime, but can be configured to your needs.

### Launchers 🚀

- [x] Distributed inference/training (`launcher=torchrun`)
- [x] Process isolation between consecutive runs (`launcher=process`)
- [x] Assert GPU devices (NVIDIA & AMD) isolation (`launcher.device_isolation=true`)

### Backends & Devices 📱

- [x] Pytorch backend for CPU (`backend=pytorch`, `backend.device=cpu`)
- [x] Pytorch backend for CUDA (`backend=pytorch`, `backend.device=cuda`)
- [ ] Pytorch backend for Habana Gaudi Processor (`backend=pytorch`, `backend.device=habana`)
- [x] OnnxRuntime backend for CPUExecutionProvider (`backend=onnxruntime`, `backend.device=cpu`)
- [x] OnnxRuntime backend for CUDAExecutionProvider (`backend=onnxruntime`, `backend.device=cuda`)
- [x] OnnxRuntime backend for ROCMExecutionProvider (`backend=onnxruntime`, `backend.device=cuda`, `backend.provider=ROCMExecutionProvider`)
- [x] OnnxRuntime backend for TensorrtExecutionProvider (`backend=onnxruntime`, `backend.device=cuda`, `backend.provider=TensorrtExecutionProvider`)
- [x] Intel Neural Compressor backend for CPU (`backend=neural-compressor`, `backend.device=cpu`)
- [x] TensorRT-LLM backend for CUDA (`backend=tensorrt-llm`, `backend.device=cuda`)
- [x] OpenVINO backend for CPU (`backend=openvino`, `backend.device=cpu`)

### Benchmarking 🏋️

- [x] Memory tracking (`benchmark.memory=true`)
- [x] Energy and efficiency tracking (`benchmark.energy=true`)
- [x] Latency and throughput tracking (`benchmark.latency=true`)
- [x] Warm up runs before inference (`benchmark.warmup_runs=20`)
- [x] Warm up steps during training (`benchmark.warmup_steps=20`)
- [x] Inputs shapes control (e.g. `benchmark.input_shapes.sequence_length=128`)
- [x] Dataset shapes control (e.g. `benchmark.dataset_shapes.dataset_size=1000`)
- [x] Prefill latency and Decoding throughput deduced from Generate and Forward pass (auto-enabled for text generation models)
- [x] Forward, Call and Generate pass kwargs control (e.g. for an LLM `benchmark.generate_kwargs.max_new_tokens=100`, for a diffusion model `benchmark.call_kwargs.num_images_per_prompt=4`)

### Backend features 🧰

- [x] "No weights" to benchmark models without downloading their weights (`backend.no_weights=true`)
- [x] Onnxruntime Quantization and AutoQuantization (`backend.quantization=true` or `backend.auto_quantization=avx2`, etc)
- [x] Onnxruntime Calibration for Static Quantization (`backend.quantization_config.is_static=true`, etc)
- [x] Onnxruntime Optimization and AutoOptimization (`backend.optimization=true` or `backend.auto_optimization=O4`, etc)
- [x] BitsAndBytes quantization scheme (`backend.quantization_scheme=bnb`, `backend.quantization_config.load_in_4bit`, etc)
- [x] GPTQ quantization scheme (`backend.quantization_scheme=gptq`, `backend.quantization_config.bits=4`, etc)
- [x] PEFT training (`backend.peft_strategy=lora`, `backend.peft_config.task_type=CAUSAL_LM`, etc)
- [x] Transformers' Flash Attention V2 (`backend.use_flash_attention_v2=true`)
- [x] Optimum's BetterTransformer (`backend.to_bettertransformer=true`)
- [x] DeepSpeed-Inference support (`backend.deepspeed_inference=true`)
- [x] Dynamo/Inductor compiling (`backend.torch_compile=true`)
- [x] Automatic Mixed Precision (`backend.amp_autocast=true`)

## Contributing 🤝

Contributions are welcome! And we're happy to help you get started. Feel free to open an issue or a pull request.
Things that we'd like to see:

- More backends (Tensorflow, TFLite, Jax, etc).
- More tests (for optimizations and quantization schemes).
- More hardware support (Habana Gaudi Processor (HPU), etc).
- Task evaluators for the most common tasks (would be great for output regression).
