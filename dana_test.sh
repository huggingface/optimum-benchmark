#!/bin/bash

# launch hydra with configs 5 times (simulating 5 days)
# possible to add --multirun to run multiple configs derived from the same base
# actually even these could be grouped in one config with multiple values for model
for i in 1 2 3 4 5; do
    python main.py --config-name bert
    python main.py --config-name bert backend=onnxruntime
    python main.py --config-name deit
    python main.py --config-name deit backend=onnxruntime
    python main.py --config-name whisper
    python main.py --config-name whisper backend=onnxruntime
done
