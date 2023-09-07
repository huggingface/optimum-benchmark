# Optimum-Benchmark (🚧 WIP 🚧)

## The Goal

A repository aiming to create a universal benchmarking utility for any model on [HuggingFace's Hub](https://huggingface.co/models) supporting [Optimum](https://github.com/huggingface/optimum)'s [inference](https://github.com/huggingface/optimum#accelerated-inference) & [training](https://github.com/huggingface/optimum#accelerated-training), optimizations & quantizations, on different backends & hardwares (OnnxRuntime, Intel Neural Compressor, OpenVINO, Habana Gaudi Processor (HPU), etc).

The experiment management and tracking is handled using [hydra](https://hydra.cc/) from the command line with minimum configuration changes and maximum flexibility (inspired by [tune](https://github.com/huggingface/tune)).

## Motivation

- Many users would want to know how their chosen model performs in terms of latency & throughput before deploying it to production.
- Many hardware vendors would want to know how their hardware performs on different models and how it compares to other hardware.
- Optimum offers a lot of hardware and backend specific optimizations that can be applied to models and improve their performance, but it might be hard to know which ones to use if you don't know a lot about your hardware. It's also hard to estimate how much these optimizations will improve the performance before training your model or downloading it from the hub and applying them.
- Benchmarks depend heavily on many factors, like the machine/hardware/os/releases etc, but these details are not put forward with the results. And that makes most of the benchmarks available or published, not very useful for decision making.
- [...]

## Features

Benchmarks:

- [x] Inference and Training benchmarks.
- [x] Latency and throughput tracking (default)
- [x] Peak memory tracking (`benchmark.memory=true`)
- [x] Energy and carbon emissions estimation (`benchmark.energy=true`)
- [x] Input shapes control (e.g. `benchmark.input_shapes.batch_size=8`)
- [x] Forward and Generation pass control (e.g. `benchmark.generate.max_new_tokens=100`, `benchmark.forward.num_images_per_prompt=4`)
- [x] Training warmup steps (`benchmark.warmup_steps=20`)

Backends:

- [x] Pytorch backend for CPU
- [x] Pytorch backend for CUDA
- [ ] Pytorch backend for Habana Gaudi Processor (HPU)
- [x] OnnxRuntime backend for CPUExecutionProvider
- [x] OnnxRuntime backend for CUDAExecutionProvider
- [x] OnnxRuntime backend for TensorrtExecutionProvider
- [x] Intel Neural Compressor backend for CPU
- [x] OpenVINO backend for CPU

Features:

- [x] Random weights initialization (practical for benchmarking big/many models withut downloading the weights).
- [x] Optimum's Quantization and AutoQuantization (Onnxruntime, OpenVINO, Intel Neural Compressor)
- [x] Optimum's Calibration for Static Quantization (Onnxruntime, OpenVINO, Intel Neural Compressor)
- [x] Optimum's Optimization and AutoOptimization (Onnxruntime)
- [x] PEFT training (Pytorch and Onnxruntime)
- [x] DDP training (Pytorch and Onnxruntime)
- [x] BitsAndBytes quantization scheme (Pytorch)
- [x] Optimum's BetterTransformer (Pytorch)
- [x] Automatic Mixed Precision (Pytorch)
- [x] GPTQ quantization scheme (Pytorch)
- [x] Dynamo compiling (Pytorch)

## Quickstart

Start by installing the required dependencies for your hardware you want to use.
For example, if you're gonna be running some GPU/CUDA benchmarks, you can install the requirements with:

```bash
python -m pip install -r gpu_requirements.txt
```

Then install the package:

```bash
python -m pip install -e .
```

You can now run a benchmark using the command line by specifying the configuration directory and the configuration name.
Both arguments are mandatory. The `config-dir` is the directory where the configuration files are stored and the `config-name` is the name of the configuration file without its `.yaml` extension.

```bash
optimum-benchmark --config-dir examples/ --config-name pytorch_bert
```

This will run the benchmark using the configuration in [`examples/pytorch_bert.yaml`](examples/pytorch_bert.yaml) and store the results in `runs/pytorch_bert`.

The result files are `inference_results.csv`, the program's logs `experiment.log` and the configuration that's been used `hydra_config.yaml`. Some other files might be generated depending on the configuration (e.g. `forward_codecarbon.csv` if `benchmark.energy=true`).

The directory for storing these results can be changed by setting `hydra.run.dir` (and/or `hydra.sweep.dir` in case of a multirun) in the command line or in the config file.

## Command-line configuration overrides

It's easy to override the default behavior of a benchmark from the command line.

```bash
optimum-benchmark --config-dir examples/ --config-name pytorch_bert model=gpt2 device=cuda:1
```

## Multirun configuration sweeps

You can easily run configuration sweeps using the `-m` or `--multirun` option. By default, configurations will be executed serially but other kinds of executions are supported with hydra's launcher plugins : `hydra/launcher=submitit`, `hydra/launcher=rays`, `hydra/launcher=joblib`, etc.

```bash
optimum-benchmark --config-dir examples --config-name pytorch_bert -m device=cpu,cuda
```

Also, for integer parameters like `batch_size`, one can specify a range of values to sweep over:

```bash
optimum-benchmark --config-dir examples --config-name pytorch_bert -m device=cpu,cuda benchmark.input_shapes.batch_size='range(1,10,step=2)'
```

## Reporting benchamrk results (WIP)

To aggregate the results of a benchmark (run(s) or sweep(s)), you can use the `optimum-report` command.

```bash
optimum-report --experiments {experiments_folder_1} {experiments_folder_2} --baseline {baseline_folder} --report-name {report_name}
```

This will create a report in the `reports` folder with the name `{report_name}`. The report will contain the results of the experiments in `{experiments_folder_1}` and `{experiments_folder_2}` compared to the results of the baseline in `{baseline_folder}` in the form of a `.csv` file, an `.svg` rich table and (a) `.png` plot(s).

You can also reuse some components of the reporting script for your use case (examples in [`examples/training-llamas`] and [`examples/running-llamas`]).

## Configurations structure

You can create custom configuration files following the [examples here](examples).
You can also use `hydra`'s [composition](https://hydra.cc/docs/0.11/tutorial/composition/) with a base configuratin ([`examples/pytorch_bert.yaml`](examples/pytorch_bert.yaml) for example) and override/define parameters.

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

Other than the [examples](examples), you can check [`tests` configuration files](tests/configs/).

## Contributing

Contributions are welcome! And we're happy to help you get started. Feel free to open an issue or a pull request.
Things that we'd like to see:

- More tests (right now we only have few tests per backend).
- Task evaluators for the most common tasks (would be great for output regression).
- More backends (Tensorflow, TFLite, Jax, etc).
- More hardware support (Habana Gaudi Processor (HPU), RadeonOpenCompute (ROCm), etc).
