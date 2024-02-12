# List of targets that are not associated with files
.PHONY:	quality style install install_dev_cpu install_dev_gpu

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .

install:
	pip install -e .

build_docker_cpu:
	docker build -f docker/cuda.dockerfile  --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t opt-bench-cpu:latest .

build_docker_cuda:
	docker build -f docker/cuda.dockerfile  --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg TORCH_CUDA=cu118 --build-arg CUDA_VERSION=11.8.0 -t opt-bench-cuda:11.8.0 . 

test_cli_cpu_neural_compressor:
	docker run \
	--rm \
	--entrypoint /bin/bash \
	--volume $(PWD):/workspace \
	--workdir /workspace \
	opt-bench-cpu:latest -c "pip install -e .[testing,neural-compressor] && pytest tests/ -k 'cli and cpu and neural_compressor' -x"

test_cli_cpu_openvino:
	docker run \
	--rm \
	--entrypoint /bin/bash \
	--volume $(PWD):/workspace \
	--workdir /workspace \
	opt-bench-cpu:latest -c "pip install -e .[testing,openvino,diffusers] && pytest tests/ -k 'cli and cpu and openvino' -x"

test_cli_cpu_onnxruntime:
	docker run \
	--rm \
	--entrypoint /bin/bash \
	--volume $(PWD):/workspace \
	--workdir /workspace \
	opt-bench-cpu:latest -c "pip install -e .[testing,onnxruntime,diffusers,timm] && pytest tests/ -k 'cli and cpu and onnxruntime' -x"

test_cli_cpu_pytorch:
	docker run \
	--rm \
	--entrypoint /bin/bash \
	--volume $(PWD):/workspace \
	--workdir /workspace \
	opt-bench-cpu:latest -c "pip install -e .[testing,diffusers,timm] && pytest tests/ -k 'cli and cpu and pytorch' -x"

test_api_cpu:
	docker run \
	--rm \
	--entrypoint /bin/bash \
	--volume $(PWD):/workspace \
	--workdir /workspace \
	opt-bench-cpu:latest -c "pip install -e .[testing,timm,diffusers,codecarbon] && pytest tests/ -k 'api and cpu' -x"

test_api_cuda:
	docker run \
	--rm \
	--gpus '"device=0,1"' \
	--entrypoint /bin/bash \
	--volume $(PWD):/workspace \
	--workdir /workspace \
	opt-bench-cuda:11.8.0 -c "pip install -e .[testing,timm,diffusers,codecarbon] && pytest tests/ -k 'api and cuda' -x"

test_api_misc:
	docker run \
	--rm \
	--entrypoint /bin/bash \
	--volume $(PWD):/workspace \
	--workdir /workspace \
	opt-bench-cpu:latest -c "pip install -e .[testing,timm,diffusers,codecarbon] && pytest tests/ -k 'api and not (cpu or cuda or rocm or tensorrt)' -x"
