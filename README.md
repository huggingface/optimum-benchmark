# inference-benchmark
A repository for benchmarking optimum's inference optimizations on different supported backends.

## Quickstart
Start by installing the required dependencies:

```
python -m pip install -r requirements.txt
```

Then, run the benchmark:

```
python src/main.py
```

The default behavior of the benchmark is determined by `configs/benchmark.yaml`.

## Command-line configuration overrides
It's easy to override the default behavior of your benchmark from the command line.

```
python src/main.py experiment_name=my-new-gpu-experiment model=bert-base-uncased backend=pytorch backend.device=cuda
```

Results (`perfs.csv` and `details.csv`) will be stored in `outputs/{experiment_name}/{experiment_id}`, along with the program logs `main.log`, the configuration that's been used `.hydra/config.yaml` and overriden parameters `.hydra/overrides.yaml`.

## Multirun configuration sweeps
You can easily run configuration sweeps using the `-m` or `--multirun` option. By default, configurations will be executed serially.

```
python src/main.py -m backend=pytorch,onnxruntime backend.device=cpu,cuda
```

## Notes

For now only `device` is supported as a multi backend parameter since it's supported by both pytorch and onnxruntime. When `backend.compile` is specified, for example, with a multirun `backenc=pytorch,onnxruntime`, it raises an error.

## TODO
- [ ] Add support for sparse inputs (zeros in the attention mask)
- [ ] Add support for onnxruntime optimizations (graph optimization, quantization, etc.)
- [ ] Add support for other model inputs (pixels, decoder_inputs, etc.)
- [ ] Add support for more metrics (memory usage, node execution time, etc.)
- [ ] Gather report data from an experiment and visualize it
- [ ] ...