# Run the tests on a local GPU

echo "Building image"
docker build -f docker/gpu_onnxruntime_training.Dockerfile -t gpu-onnxruntime-training .

echo "Running tests within the container"
docker run --rm --gpus all --workdir=/workspace/optimum-benchmark gpu-onnxruntime-training \
pytest -k "(cuda or tensorrt) and onnxruntime_training"