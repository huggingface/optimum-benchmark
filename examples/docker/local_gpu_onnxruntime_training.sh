# Run the tests on a local GPU

echo "Building image"
docker build -f examples/docker/gpu_onnxruntime_training.Dockerfile -t gpu-onnxruntime-training .

echo "Running tests within the container"
docker run --rm --gpus all --workdir=/workspace/optimum-benchmark gpu-onnxruntime-training