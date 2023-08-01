# Optimum-Benchmark (🚧 WIP 🚧)

## The Goal

A repository aiming to create a benchmarking utility for any model on [HuggingFace's Hub](https://huggingface.co/models) supporting [Optimum](https://github.com/huggingface/optimum)'s [inference](https://github.com/huggingface/optimum#accelerated-inference) & [training](https://github.com/huggingface/optimum#accelerated-training), optimizations & quantizations, on different backends & hardwares (OnnxRuntime, Intel Neural Compressor, OpenVINO, Habana Gaudi Processor (HPU), etc).

The experiment management and tracking is handled by [hydra](https://hydra.cc/) using the command line with minimum configuration changes and maximum flexibility (inspired from [tune](https://github.com/huggingface/tune)).

## Motivation

- Many users would want to know how their chosen model performs (latency & throughput) before deploying it to production.
- Many hardware vendors would want to know how their hardware performs on different models and how it compares to others.
- Optimum offers a lot of optimizations that can be applied to models and improve their performance, but it's hard to know which ones to use if you don't know a lot about your hardware. It's also hard to estimate how much these optimizations will improve the performance before training your model or downloading it from the hub and optimizing it.
- Benchmarks depend heavily on many factors, like the machine/hardware/os/releases/etc but most of this information is not put forward with the results. And that makes most of the benchmarks available today, not very useful for decision making.
- [...]

## Features

General:

- [x] Latency and throughput tracking (default behavior)
- [x] Peak memory tracking (`benchmark.memory=true`)
- [x] Symbolic Profiling (`benchmark.profile=true`)
- [x] Input shapes control (e.g. `benchmark.input_shapes.batch_size=8`)
- [x] Random weights initialization (`backend.no_weights=true` support depends on backend)

Inference:

- [x] Pytorch backend for CPU
- [x] Pytorch backend for CUDA
- [ ] Pytorch backend for Habana Gaudi Processor (HPU)
- [x] OnnxRuntime backend for CPUExecutionProvider
- [x] OnnxRuntime backend for CUDAExecutionProvider
- [x] Intel Neural Compressor backend for CPU
- [x] OpenVINO backend for CPU

Optimizations:

- [x] Pytorch's Automatic Mixed Precision
- [x] Optimum's BetterTransformer
- [x] Optimum's Optimization and AutoOptimization
- [x] Optimum's Quantization and AutoQuantization
- [x] Optimum's Calibration for Static Quantization
- [x] BitsAndBytes' quantization

## Quickstart

Start by installing the required dependencies depending on your hardware and the backends you want to use.
For example, if you're gonna be running some GPU benchmarks, you can install the requirements with:

```bash
python -m pip install -r gpu_requirements.txt
```

Then install the package:

```bash
python -m pip install -e .
```

You can now run a benchmark using the command line by specifying the configuration directory and the configuration name.
Both arguments are mandatory. The `config-dir` is the directory where the configuration files are stored and the `config-name` is the name of the configuration file without the `.yaml` extension.

```bash
optimum-benchmark --config-dir examples --config-name pytorch
```

This will run the benchmark on the `pytorch` backend and `cpu` device. Resultq will be stored in `runs/bert_baseline`.

The result files are `inference_results.csv` and `profiling_results.csv` in case profiling is enabled (`benchmark.profile=true`), in addition to the program's logs `main.log` and the configuration that's been used `hydra_config.yaml`

The directory for storing these results can be changed using the `hydra.run.dir` (and/or `hydra.sweep.dir`) in the command line or in the config file (see [`base_config.yaml`](examples/base_config.yaml)).

## Command-line configuration overrides

It's easy to override the default behavior of a benchmark from the command line.

```bash
optimum-benchmark --config-dir examples --config-name pytorch model=gpt2 device=cuda:1
```

## Multirun configuration sweeps

You can easily run configuration sweeps using the `-m` or `--multirun` option. By default, configurations will be executed serially but other kinds of executions are supported with hydra's launcher plugins : `hydra/launcher=submitit`, `hydra/launcher=rays`, etc.

```bash
optimum-benchmark --config-dir examples --config-name pytorch -m device=cpu,cuda
```

Also, for integer parameters like `batch_size`, one can specify a range of values to sweep over:

```bash
optimum-benchmark --config-dir examples --config-name pytorch -m device=cpu,cuda benchmark.input_shapes.batch_size='range(1,10,step=2)'
```

## Reporting benchamrk results (WIP)

To aggregate the results of a benchmark (run(s) or sweep(s)), you can use the `optimum-report` command.

```bash
optimum-report --experiments {experiments_folder_1} {experiments_folder_2} --baseline {baseline_folder} --report-name {report_name}
```

This will create a report in the `reports` folder with the name `{report_name}`. The report will contain the results of the experiments in `{experiments_folder_1}` and `{experiments_folder_2}` compared to the results of the baseline in `{baseline_folder}`. The baseline is optional.

## Configurations structure

You can create custom configuration files following the [examples here](examples).
The easiest way to do so is by using `hydra`'s [composition](https://hydra.cc/docs/0.11/tutorial/composition/) with a base configuratin [`examples/base_config.yaml`](examples/base_config.yaml).

To create a configuration that uses a `wav2vec2` model and `onnxruntime` backend, it's as easy as:

```yaml
defaults:
  - base_config
  - _self_
  - override backend: onnxruntime

experiment_name: onnxruntime_wav2vec2

model: bookbot/distil-wav2vec2-adult-child-cls-37m
device: cpu
```

Some examples are provided in the [`examples`](examples) folder for different backends and models.

## TODO

- [x] Add support for any kind of input (text, audio, image, etc.)
- [x] Add support for onnxruntime backend
- [x] Add support for optimum quantization
- [x] Add support for omptimum graph optimizations
- [x] Add support for static quantization + calibration.
- [x] Add support for profiling nodes/kernels execution time.
- [x] Add experiments aggregator to report on data from different runs/sweeps.
- [x] Add support for sweepers latency optimization (optuna, nevergrad, etc.)
- [x] Add support for more metrics (memory usage, node execution time, etc.)
- [x] Migrate configuration management to be handled solely by config store.
- [x] Add Dana client to send results to the dashboard [(WIP)](https://github.com/huggingface/dana)
- [x] Make a consistent reporting utility.
- [ ] Add Pydantic for schema validation.
- [ ] Add support for sparse inputs.
- [ ] ...
