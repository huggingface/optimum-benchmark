# optimum-benchmarks

A repository for benchmarking optimum's inference and training optimizations on different supported backends.
The experiment management and tracking is handled by [hydra](https://hydra.cc/) and inspired from [tune](https://github.com/huggingface/tune).

## Quickstart

Start by installing the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Then, to run the default `bert` benchmark:

```bash
python main.py --config-name bert
```

Who's behavior is determined by the [`config/bert.yaml`](configs/bert.yaml).

## Command-line configuration overrides

It's easy to override the default behavior of a benchmark from the command line.

```
python main.py --config-name whisper backend=onnxruntime device=cpu
```

Experiment results `inference_results.csv` (and `profiling_results.csv` in case profiling is enabled) will be stored in `runs/${backend.name}_${device}/${experiment_name}`, along with the program logs `main.log` and the configuration that's been used `hydra_config.yaml`. The directory for storing these results can be changed using the `hydra.run.dir` in the command line or in the configuration file (see [`hydra_base.yaml`](configs/hydra_base.yaml)).

## Multirun configuration sweeps

You can easily run configuration sweeps using the `-m` or `--multirun` option. By default, configurations will be executed serially but other kinds of executions are supported with hydra's launcher plugins : `hydra/launcher=submitit`, `hydra/launcher=rays`, etc.

```bash
python main.py --config-name bert -m backend=pytorch,onnxruntime device=cpu,cuda
```

Also, for integer parameters like `batch_size`, one can specify a range of values to sweep over:

```bash
python main.py --config-name bert -m backend=pytorch,onnxruntime device=cpu,cuda benchmark.input.batch_size='range(1,10,step=2)'
```

Other features like intervals and log scaled ranges of values are also supported through sweeper plugins: `hydra/sweeper=optuna`, `hydra/sweeper=nevergrad`, etc.

## Reporting experiment results

To aggregate the results of an experiment (run(s) or sweep(s)), you can use the `reporter.py` script:

```bash
python reporter.py --experiments-folder {folder_path}
```

This will generate `inference_report.csv` in the specified experiments directory which contains the aggregated results of all the runs/sweeps with their corresponding configurations.

To compare these results with a baseline, you can use the optional `--baseline-folder` option:

```bash
python reporter.py --experiments-folder {folder_path} --baseline-folder {folder_path}
```

The console outputs will be something like this:
<img src='rich-benchmark.png' alt='rich-benchmark-table' style='display:block;margin-left:auto;margin-right:auto;'>

## Configurations structure

You can create custom configuration files following the examples in `configs` directory.
The easiest way to do so is by using `hydra`'s [composition](https://hydra.cc/docs/0.11/tutorial/composition/).

The base configuration is [`configs/base_experiment.yaml`](configs/base_experiment.yaml).
For example, to create a configuration that uses a `wav2vec2` model and `onnxruntime` backend, it's as easy as:

```yaml
defaults:
  - hydra_base
  - _self_
  - override backend: onnxruntime

# experiment name can be set or inferred from pramaeters
model: bookbot/distil-wav2vec2-adult-child-cls-37m
device: cpu

experiment_name: onnxruntime_wav2vec2
```

This is especially useful for creating sweeps, where the cli commands become too long. An example is provided in [`configs/optuna.yaml`](configs/optuna.yaml) for an exhaustive sweep over all possible cominations of `onnxruntime`'s graph optimizations (level, layer fusions, etc.) and quantizations (operator, weights, etc.). The command to run it is:

```bash
python main.py --config-name optuna -m
```

But in this example in particule we don't use the basic sweeper (that's used for testing all combinations) but rather a custom one that leverages [optuna](https://optuna.org/) to find the best combination in `n_trials` run, reducing the latency with bayesian optimization (isn't that cool?).

At the end of it you get an additional `optimization_results.yaml` file that contains the best combination of parameters found by optuna.

## TODO

- [x] Add support for any kind of input (text, audio, image, etc.)
- [x] Add support for onnxruntime backend
- [x] Add support for omptimum graph optimizations
- [x] Add support for optimum quantization
- [x] Add experiments aggregator to report on data from different runs/sweeps.
- [x] Add support for sweepers latency optimization (optuna, nevergrad, etc.)
- [x] Add support for profiling nodes/kernels execution time.
- [x] add Dana client to send results to the dashboard [(WIP)](https://github.com/IlyasMoutawwakil/optimum-dana)
- [x] Add support for more metrics (memory usage, node execution time, etc.)

- [ ] Migrate configuration management to be handled solely by dataclasses.
- [ ] Add Pydantic for schema validation.
- [ ] Add support for quantization calibration.
- [ ] Add support for sparse inputs (zeros in the attention mask)
- [ ] ...
