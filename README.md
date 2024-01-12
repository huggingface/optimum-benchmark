# Optimum-Benchmark

Optimum-Benchmark is a unified multi-backend utility for benchmarking `transformers`, `diffusers`, `peft` and `timm` models with [Optimum](https://github.com/huggingface/optimum)'s optimizations & quantization schemes, for [inference](https://github.com/huggingface/optimum#accelerated-inference) & [training](https://github.com/huggingface/optimum#accelerated-training), on different backends & hardwares (OnnxRuntime, Intel Neural Compressor, OpenVINO, Habana Gaudi Processor (HPU), AMD Instinct GPUs, etc).

The experiment management and tracking is handled using [hydra](https://hydra.cc/) which allows for simple cli with minimum configuration changes and maximum flexibility (inspired by [tune](https://github.com/huggingface/tune)).

## Motivation

- Hardware vendors wanting to know how their hardware performs compared to others on the same models.
- HF users wanting to know how their chosen model performs in terms of latency, throughput, memory usage, energy consumption, etc.
- Experimenting with Optimum and Transformers' arsenal of optimizations & quantization schemes that can be applied to models and improve their computational/memory/energy efficiency.
- [...]

## Features

`optimum-benchmark` allows you to run benchmarks with no code and minimal user input, just specify:

- The type of device (e.g. `cuda`).
- The launcher to use (e.g. `process`).
- The type of benchmark (e.g. `training`)
- The backend to run on (e.g. `onnxruntime`).
- The model name or path (e.g. `bert-base-uncased`)
- And optionally, the model's task (e.g. `text-classification`).

Everything else is either optional or inferred from the model's name or path.

### Supported Backends/Devices

- [x] Pytorch backend for CPU (Intel, AMD, ARM, etc)
- [x] Pytorch backend for CUDA (NVIDIA and AMD GPUs)
- [ ] Pytorch backend for Habana Gaudi Processor (HPU)
- [x] OnnxRuntime backend for CPUExecutionProvider
- [x] OnnxRuntime backend for CUDAExecutionProvider
- [ ] OnnxRuntime backend for ROCMExecutionProvider
- [x] OnnxRuntime backend for TensorrtExecutionProvider
- [x] Intel Neural Compressor backend for CPU
- [x] OpenVINO backend for CPU

### Launcher features

- [x] Process isolation between consecutive runs (`launcher=process`)
- [x] Assert devices (NVIDIA & AMD GPUs) isolation (`launcher.device_isolation=true`)
- [x] Distributed inference/training (`launcher=torchrun`, `launcher.n_proc_per_node=2`, etc)

### Benchmark features

- [x] Memory tracking (`benchmark.memory=true`)
- [x] Latency and throughput tracking of forward pass (default)
- [x] Warm up runs before inference (`benchmark.warmup_runs=20`)
- [x] Warm up steps during training (`benchmark.warmup_steps=20`)
- [x] Energy and carbon emissions tracking (`benchmark.energy=true`)
- [x] Inputs shapes control (e.g. `benchmark.input_shapes.sequence_length=128`)
- [x] Dataset shapes control (e.g. `benchmark.dataset_shapes.dataset_size=1000`)
- [x] Latancy and throughput tracking of generation pass (auto-enabled for generative models)
- [x] Prefill latency and Decoding throughput deduced from generation and forward pass (auto-enabled for generative models)
- [x] Forward and Generation pass control (e.g. for an LLM `benchmark.generate_kwargs.max_new_tokens=100`, for a diffusion model `benchmark.forward_kwargs.num_images_per_prompt=4`)

### Backend features

- [x] Random weights initialization (`backend.no_weights=true` for fast model instantiation without downloading weights)
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

## Quickstart

### Installation

You can install `optimum-benchmark` using pip:

```bash
python -m pip install git+https://github.com/huggingface/optimum-benchmark.git
```

or by cloning the repository and installing it in editable mode:

```bash
git clone https://github.com/huggingface/optimum-benchmark.git && cd optimum-benchmark && python -m pip install -e .
```

Depending on the backends you want to use, you might need to install some extra dependencies:

- OpenVINO: `pip install optimum-benchmark[openvino]`
- OnnxRuntime: `pip install optimum-benchmark[onnxruntime]`
- OnnxRuntime-GPU: `pip install optimum-benchmark[onnxruntime-gpu]`
- Intel Neural Compressor: `pip install optimum-benchmark[neural-compressor]`
- Text Generation Inference: `pip install optimum-benchmark[text-generation-inference]`

You can now run a benchmark using the command line by specifying the configuration directory and the configuration name. Both arguments are mandatory for `hydra`. `--config-dir` is the directory where the configuration files are stored and `--config-name` is the name of the configuration file without its `.yaml` extension.

```bash
optimum-benchmark --config-dir examples/ --config-name pytorch_bert
```

This will run the benchmark using the configuration in [`examples/pytorch_bert.yaml`](examples/pytorch_bert.yaml) and store the results in `runs/pytorch_bert`.

The result files are `inference_results.csv`, the program's logs `experiment.log` and the configuration that's been used `hydra_config.yaml`. Some other files might be generated depending on the configuration (e.g. `forward_codecarbon.csv` if `benchmark.energy=true`).

The directory for storing these results can be changed by setting `hydra.run.dir` (and/or `hydra.sweep.dir` in case of a multirun) in the command line or in the config file.

## Command-line configuration overrides

It's easy to override the default behavior of a benchmark from the command line.

```bash
optimum-benchmark --config-dir examples/ --config-name pytorch_bert model=gpt2 device=cuda
```

## Multirun configuration sweeps

You can easily run configuration sweeps using the `-m` or `--multirun` option. By default, configurations will be executed serially but other kinds of executions are supported with hydra's launcher plugins : `=submitit`, `hydra/launcher=rays`, etc.
Note that the hydra launcher `hydra/launcher` is different than our own `launcher`, specifically `hydra/launcher` can only be used in `--multirun` mode, and will only handle the inter-run behavior.

```bash
optimum-benchmark --config-dir examples --config-name pytorch_bert -m device=cpu,cuda
```

Also, for integer parameters like `batch_size`, one can specify a range of values to sweep over:

```bash
optimum-benchmark --config-dir examples --config-name pytorch_bert -m device=cpu,cuda benchmark.input_shapes.batch_size='range(1,10,step=2)'
```

## Configurations structure

You can create custom configuration files following the [examples here](examples).
You can also use `hydra`'s [composition](https://hydra.cc/docs/0.11/tutorial/composition/) with a base configuration ([`examples/pytorch_bert.yaml`](examples/pytorch_bert.yaml) for example) and override/define parameters.

To create a configuration that uses a `wav2vec2` model and `onnxruntime` backend, it's as easy as:

```yaml
defaults:
  - pytorch_bert
  - _self_
  - override backend: onnxruntime

experiment_name: onnxruntime_wav2vec2
model: bookbot/distil-wav2vec2-adult-child-cls-37m
device: cpu
```

Other than the [examples](examples), you can also check [`tests`](tests/configs/).

## Contributing

Contributions are welcome! And we're happy to help you get started. Feel free to open an issue or a pull request.
Things that we'd like to see:

- More backends (Tensorflow, TFLite, Jax, etc).
- More hardware support (Habana Gaudi Processor (HPU), etc).
- More tests (right now we only have few tests per backend).
- Task evaluators for the most common tasks (would be great for output regression).
