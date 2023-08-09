# Run the tests on a local GPU

echo "Building image"
docker build -f docker/gpu.Dockerfile -t optimum-benchmark-gpu .

echo "Running tests within the container"
docker run --rm --gpus all --workdir=/workspace/optimum-benchmark optimum-benchmark-gpu \
pytest -k "(cuda or tensorrt) and not onnxruntime_training"