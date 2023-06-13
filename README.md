# Optimum-Benchmark

## The Goal

A repository aiming to create a benchmarking utility for any model on [HuggingFace's Hub](https://huggingface.co/models) supporting [Optimum](https://github.com/huggingface/optimum)'s [inference](https://github.com/huggingface/optimum#accelerated-inference) and [training](https://github.com/huggingface/optimum#accelerated-training) optimizations on different backends and hardware (OnnxRuntime, Intel Neural Compressor, OpenVINO, Habana Gaudi Processor (HPU), etc.).

The experiment management and tracking is handled by [hydra](https://hydra.cc/) using the command line with minimum mandatory configuration changes and maximum flexibility (inspired from [tune](https://github.com/huggingface/tune))

## Motivation

- Many users would want to know how their chosen model performq (latency/throughput) before deploying it to production.
- Many hardware vendors would want to know how their hardware performs on different models and how it compares to others.
- Optimum offers a lot of optimizations that can be applied to models and improve their performance, but it's hard to know which ones to use if you don't know a lot about your hardware. It's also hard to estimate how much these optimizations will improve the performance before trying them out.
- Benchmarks depend heavily on many factors, like the machine/hardware/os/releases/etc. And that makes most of the benchmarks available today, not very useful for decision making.
- [...]

## Features

Inference:

- [x] Pytorch backend for cpu
- [x] Pytorch backend for cuda
- [ ] Pytorch backend for hpu
- [x] OnnxRuntime backend for cpu
- [x] OnnxRuntime backend for cuda
- [ ] OnnxRuntime backend for tensorrt
- [ ] Intel Neural Compressor backend
- [ ] OpenVINO backend

Optimizations:

- [x] Pytorch's FP16
- [x] Optimum's BetterTransformer
- [x] Optimum's Optimization and AutoOptimization
- [x] Optimum's Quantization and AutoQuantization
- [ ] Optimum's Calibration for Static Quantization

## Quickstart

Start by installing the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Then copy `examples/bert.yaml` to `configs/bert.yaml` and run with:

```bash
python main.py --config-name bert
```

This will run the benchmark on the default backend (`pytorch`) and device (`cuda`) and store the results in `runs/bert_baseline`.

Only key parameters are overriden/defined in the config file which inherits from `configs/hydra_base.yaml` where most of the experiment's logic is defined.

The result files are `inference_results.csv` and `profiling_results.csv` in case profiling is enabled (`benchmark.profile=true`), in addition to the program's logs `main.log` and the configuration that's been used `hydra_config.yaml`

The directory for storing these results can be changed using the `hydra.run.dir` (and/or `hydra.sweep.dir`) in the command line or in the config file (see [`hydra_base.yaml`](configs/hydra_base.yaml)).

## Command-line configuration overrides

It's easy to override the default behavior of a benchmark from the command line.

```
python main.py --config-name bert backend=onnxruntime device=cpu
```

## Multirun configuration sweeps

You can easily run configuration sweeps using the `-m` or `--multirun` option. By default, configurations will be executed serially but other kinds of executions are supported with hydra's launcher plugins : `hydra/launcher=submitit`, `hydra/launcher=rays`, etc.

```bash
python main.py --config-name bert -m backend=pytorch,onnxruntime device=cpu,cuda
```

Also, for integer parameters like `batch_size`, one can specify a range of values to sweep over:

```bash
python main.py --config-name bert -m backend=pytorch,onnxruntime device=cpu,cuda benchmark.batch_size='range(1,10,step=2)'
```

Other features like intervals and log scaled ranges of values are also supported through sweeper plugins: `hydra/sweeper=optuna`, `hydra/sweeper=nevergrad`, etc.

## Reporting benchamrk results (WIP)

To aggregate the results of a benchmark (run(s) or sweep(s)), you can use the `reporter.py` script:

```bash
python reporter.py --baseline {baseline_folder} --experiments {experiments_folder_1} {experiments_folder_2} ...
```

Where baseline is optional (used to compute speedups) and experiments are the folders containing the results of the runs/sweeps you want to aggregate.

The script will generate a few reporting files : a csv report (`inference_report.csv`), a rich table (`rich_table.svg`) and some plots (`forward_throughput.png` and `generate_throughput.png` when possible).

These files will be stored in `reports/${device}_${batch_size}` (or `reports/${device}_${batch_size}_${new_tokens}` if generation is supported for the model).

Check Whisper's example in [`examples/whisper/`](examples/whisper/) for a full example.

## Configurations structure

You can create custom configuration files following the [examples here](examples).
The easiest way to do so is by using `hydra`'s [composition](https://hydra.cc/docs/0.11/tutorial/composition/) with a base configuratin [`configs/hydra_base.yaml`](configs/hydra_base.yaml).

To create a configuration that uses a `wav2vec2` model and `onnxruntime` backend, it's as easy as:

```yaml
defaults:
  - hydra_base
  - _self_
  - override backend: onnxruntime

experiment_name: onnxruntime_wav2vec2

model: bookbot/distil-wav2vec2-adult-child-cls-37m
device: cpu
```

This is especially useful for creating sweeps, where the cli commands become too long.

An example is provided in [`examples/whisper_auto_opt+qnt.yaml`](examples/whisper_auto_opt+qnt.yaml) for an exhaustive sweep over all possible cominations of `optimum`'s AutoOptimizations and AutoQuantizations on CPU.

## TODO

- [x] Add support for any kind of input (text, audio, image, etc.)
- [x] Add support for onnxruntime backend
- [x] Add support for omptimum graph optimizations
- [x] Add support for optimum quantization
- [x] Add experiments aggregator to report on data from different runs/sweeps.
- [x] Add support for sweepers latency optimization (optuna, nevergrad, etc.)
- [x] Add support for profiling nodes/kernels execution time.
- [x] Add support for more metrics (memory usage, node execution time, etc.)
- [x] Migrate configuration management to be handled solely by config store (practical or not?)
- [ ] Add Pydantic for schema validation.
- [ ] Find a way to seperate where experiments are stored from the configuration files (shouldn't be too long and should follow some kind of convention).
- [ ] Make a consistent reporting utility.
- [ ] Add support for static quantization + calibration.
- [ ] Add support for sparse inputs.
- [ ] ...
- [x] add Dana client to send results to the dashboard [(WIP)](https://github.com/IlyasMoutawwakil/optimum-dana)
