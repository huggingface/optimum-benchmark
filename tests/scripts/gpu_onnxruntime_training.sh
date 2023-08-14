docker run --rm --gpus all -v $(pwd):/workspace/optimum-benchmark --workdir=/workspace/optimum-benchmark gpu-onnxruntime-training /bin/bash -c "\
pip install -r gpu_onnxruntime_training_requirements.txt \
&& pip install -e .[test] \
&& pytest -k '(cuda or tensorrt) and onnxruntime_training' \
&& rm -rf ./*"