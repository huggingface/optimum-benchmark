# Optimum-Benchmark x LLaMAs x GPTQ

A set of benchmarks on Meta's LLaMA2's inference.

## Setup

You will need to install these quantization packages:

```bash
pip install auto-gptq 
```

## Running

Then run these commands from this directory:

```bash
optimum-benchmark --config-dir configs/ --config-name fp16 --multirun
optimum-benchmark --config-dir configs/ --config-name bnb-4bit --multirun
```

This will create a folder called `experiments` with the results of the benchmarks with an inference `batch_size` ranging from 1 to 16 and an input `sequence_length` (prompt size) of 256.

## Reporting

To create a report run:

```bash
python report.py -e experiments -m allocated
```

Which will create some quick reporting artifacts like a `full_report.csv`, `short_report.csv`, some plots and a `rich_table.svg`.

`-e` is the experiments folder from which to read the results.
`-r` is the report folder to which to write the resulting artifacts.
`-m` is the memory type to use for the reporting. It can be `used`, `allocated` or `reserved`.


## Results

### On A100-80GB

<p align="center">
<img src="artifacts/A100-80GB/forward_latency_plot.png" alt="latency_plot" width="60%"/>
</p>

<p align="center">
<img src="artifacts/A100-80GB/generate_throughput_plot.png" alt="throughput_plot" width="60%"/>
</p>

<p align="center">
<img src="artifacts/A100-80GB/forward_memory_plot.png" alt="memory_plot" width="60%"/>
</p>

<p align="center">
<img src="artifacts/A100-80GB/generate_memory_plot.png" alt="memory_plot" width="60%"/>
</p>

<p align="center">
<img src="artifacts/A100-80GB/rich_table.svg" alt="rich_table" width="90%"/>
</p>
