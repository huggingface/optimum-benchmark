# Examples

This folder contains examples of config files for different models and use cases.

To use a config file you should first copy it to the [configs](../config) folder and then run [`main.py`](../main.py) with it in the `--config-path` option.

If you want to run some sweep logic in your config file, don't forget to add the `-m` or `--multirun` option.

## Whisper

In the folder [whisper](whisper) you can find config files, scripts, results and reports for a benchmark I ran on [OpenAI's Whisper Base Model](https://huggingface.co/openai/whisper-base).
If you want reproduce the benchmark on your own hardware, just copy the config files to the [configs](../config) folder, copy the shel scripts to the root of the repo ([`whisper_benchmark.sh`](whisper/whisper_benchmark.sh) and [`whisper_report.sh`](whisper/whisper_report.sh)) and then run them.