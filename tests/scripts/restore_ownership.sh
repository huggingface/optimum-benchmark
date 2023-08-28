# restore ownership
docker run \
--rm \
--entrypoint /bin/bash \
--env HOST_UID=`id -u` \
--env HOST_GID=`id -g` \
--volume $(pwd):/workspace/optimum-benchmark \
--workdir /workspace/optimum-benchmark \
ubuntu \
-c 'chown -R ${HOST_UID}:${HOST_GID} /workspace/optimum-benchmark'