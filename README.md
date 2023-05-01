# inference-benchmark
A repository for benchmarking optimum's inference optimizations on different supported backends.

## CLI Usage
To run a benchmark, use the following command:
```
python src/main.py --multirun backend=pytorch,onnxruntime backend.device=cpu,cuda
```

- for now only device is supported as a multi backend parameter since it's only supported by pytorch and not onnxruntime. When specified with `backenc=pytorch,onnxruntime` it raises an error when running onnxruntime backend.

## TODO
- [ ] Add support for sparse inputs (zeros in the attention mask)
- [ ] Add support for onnxruntime optimizations (graph optimization, quantization, etc.)
- [ ] Add support for other model inputs (pixels, decoder_inputs, etc.)
- [ ] ...