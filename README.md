# inference-benchmark
A repository for benchmarking optimum's inference optimizations on different supported backends.
The configuration management is handled by [hydra](https://hydra.cc/) and based on [tune](https://github.com/huggingface/tune).

## Quickstart
Start by installing the required dependencies:

```
python -m pip install -r requirements.txt
```

Then, run the benchmark:

```
python main.py
```

The default behavior of the benchmark is determined by `configs/benchmark.yaml`.

## Command-line configuration overrides
It's easy to override the default behavior of your benchmark from the command line.

```
python main.py experiment_name=my-new-gpu-experiment model=bert-base-uncased backend=pytorch backend.device=cuda
```

Results (`perfs.csv` and `details.csv`) will be stored in `outputs/{experiment_name}/{experiment_datetime_id}`, along with the program logs `main.log`, the configuration that's been used `.hydra/config.yaml` and overriden parameters `.hydra/overrides.yaml`.

## Multirun configuration sweeps
You can easily run configuration sweeps using the `-m` or `--multirun` option. By default, configurations will be executed serially.

```
python main.py -m backend=pytorch,onnxruntime backend.device=cpu,cuda
```

Moreover, for integer parameters like `batch_size`, one can specify a range of values to sweep over:

```
python main.py -m backend=pytorch,onnxruntime backend.device=cpu,cuda batch_size='range(1, 10, step=2)'
```

Other features like log scaling a range of values are also supported through plugins.

## Notes

For now, sweeps can only run over parameters that are supported by both pytorch and onnxruntime (like `device`, `num_threads`, etc). When, for example, `backend.torch_compile` is specified on a multi backend sweep (i.e. `backend=pytorch,onnxruntime`), it raises an error.

## TODO
- [x] Add support for sparse inputs (zeros in the attention mask)
- [ ] Add support for onnxruntime optimizations (graph optimization, quantization, etc.)
- [ ] Add support for other model inputs (pixels, decoder_inputs, etc.)
- [ ] Add support for more metrics (memory usage, node execution time, etc.)
- [ ] Gather report data from an experiment and visualize it
- [ ] ...