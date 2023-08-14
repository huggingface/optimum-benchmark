docker run --rm --gpus all -v $(pwd):/workspace/optimum-benchmark --workdir=/workspace/optimum-benchmark optimum-benchmark-gpu /bin/bash -c "\
pip install -r gpu_requirements.txt \
&& pip install -e .[test] \
&& pytest -k '(cuda or tensorrt) and not onnxruntime_training' \
&& rm -rf .pytest_cache"