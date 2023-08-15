# install optimum-benchmark and run tests
docker run \
--rm \
--entrypoint /bin/bash \
--gpus all \
--volume $(pwd):/workspace/optimum-benchmark \
--workdir /workspace/optimum-benchmark \
optimum-benchmark-ort-training \
-c "pip install -e .[test] && pytest -k 'cuda_onnxruntime_training' -x"

# restore ownership
docker run \
--rm \
--entrypoint /bin/bash \
--env HOST_UID=`id -u` \
--env HOST_GID=`id -g` \
--volume $(pwd):/workspace/optimum-benchmark \
--workdir /workspace/optimum-benchmark \
optimum-benchmark-ort-training \
-c 'chown -R ${HOST_UID}:${HOST_GID} /workspace/optimum-benchmark'