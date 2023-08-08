# Run the tests on a local GPU

echo "Building image"
docker build -f examples/docker/gpu.dockerfile -t optimum-benchmark-gpu .

echo "Running tests within the container"
docker run --rm --gpus all --workdir=/workspace/optimum-benchmark optimum-benchmark-gpu