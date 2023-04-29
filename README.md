# inference-benchmark
A repository for benchmarking optimum's inference optimizations on transformers.

Hopefully a consistent multi backend and multi configuration benchmarking utility.

## CLI Usage
To run a benchmark, use the following command:
```
python src/main.py --multirun backend=pytorch backend.device=cpu,cuda backend.use_compile=false,true
```