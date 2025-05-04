# Makefile for Optimum-Benchmark

PWD := $(shell pwd)
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .

install:
	pip install -e .

## Build docker

build_cpu_image:
	docker build -t optimum-benchmark:latest-cpu -f docker/cpu/Dockerfile .
	# docker build --build-arg IMAGE=optimum-benchmark:latest-cpu --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-cpu docker/unroot

build_cuda_image:
	docker build -t optimum-benchmark:latest-cuda -f docker/cuda/Dockerfile .
	# docker build --build-arg IMAGE=optimum-benchmark:latest-cuda --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-cuda docker/unroot

build_cuda_ort_image:
	docker build -t optimum-benchmark:latest-cuda-ort -f docker/cuda-ort/Dockerfile .
	# docker build --build-arg IMAGE=optimum-benchmark:latest-cuda-ort --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-cuda-ort docker/unroot

build_rocm_image:
	docker build -t optimum-benchmark:latest-rocm -f docker/rocm/Dockerfile .
	# docker build --build-arg IMAGE=optimum-benchmark:latest-rocm --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-rocm docker/unroot

# Run docker

run_cpu_container:
	docker run \
	-it \
	--rm \
	--pid host \
	--volume $(PWD):/optimum-benchmark \
	--workdir /optimum-benchmark \
	optimum-benchmark:latest-cpu

run_cuda_container:
	docker run \
	-it \
	--rm \
	--pid host \
	--gpus all \
	--shm-size 64G \
	--volume $(PWD):/optimum-benchmark \
	--workdir /optimum-benchmark \
	optimum-benchmark:latest-cuda

run_cuda_ort_container:
	docker run \
	-it \
	--rm \
	--pid host \
	--gpus all \
	--shm-size 64G \
	--volume $(PWD):/optimum-benchmark \
	--workdir /optimum-benchmark \
	optimum-benchmark:latest-cuda-ort

run_rocm_container:
	docker run \
	-it \
	--rm \
	--shm-size 64G \
	--device /dev/kfd \
	--device /dev/dri \
	--volume $(PWD):/optimum-benchmark \
	--workdir /optimum-benchmark \
	optimum-benchmark:latest-rocm

## Install extras

install_api_misc:
	pip install -e .[testing,timm,diffusers,peft,codecarbon]

install_api_cpu:
	pip install -e .[testing,timm,diffusers,peft,codecarbon]

install_api_cuda:
	pip install -e .[testing,timm,diffusers,peft,codecarbon]

install_api_rocm:
	pip install -e .[testing,timm,diffusers,peft,codecarbon]

install_cli_misc:
	pip install -e .[testing,timm,diffusers,peft,codecarbon]

install_cli_cpu_pytorch:
	pip install -e .[testing,peft,timm,diffusers]

install_cli_cpu_openvino:
	pip install -e .[testing,peft,timm,diffusers,openvino]

install_cli_cpu_onnxruntime:
	pip install -e .[testing,peft,timm,diffusers,onnxruntime]

install_cli_cpu_neural_compressor:
	pip install -e .[testing,peft,timm,diffusers,neural-compressor]

install_cli_cuda_pytorch:
	pip install -e .[testing,timm,diffusers,peft,autoawq,gptqmodel,bitsandbytes,deepspeed]

install_cli_rocm_pytorch:
	pip install -e .[testing,timm,diffusers,peft,autoawq,gptqmodel]

install_cli_cuda_torch_ort:
	pip install -e .[testing,timm,diffusers,peft,torch-ort,deepspeed]

install_cli_cuda_onnxruntime:
	pip install -e .[testing,timm,diffusers,peft,onnxruntime-gpu]

# Run tests

test_api_misc:
	pytest -s -k "api and not (cpu or cuda or mps)"

test_api_cpu:
	pytest -s -k "api and cpu"

test_api_cuda:
	pytest -s -k "api and cuda"

test_api_rocm:
	pytest -s -k "api and cuda"

test_cli_misc:
	pytest -s -k "cli and not (cpu or cuda or mps)"

test_cli_cpu_pytorch:
	pytest -s -k "cli and cpu and pytorch"

test_cli_cpu_openvino:
	pytest -s -k "cli and cpu and openvino"

test_cli_cpu_onnxruntime:
	pytest -s -k "cli and cpu and onnxruntime"

test_cli_cpu_neural_compressor:
	pytest -s -k "cli and cpu and neural-compressor"

test_cli_cuda_onnxruntime:
	pytest -s -k "cli and cuda and onnxruntime"

test_cli_cuda_vllm:
	pytest -s -k "cli and cuda and vllm"

test_cli_cuda_pytorch_multi_gpu:
	pytest -s -k "cli and cuda and pytorch and (dp or ddp or device_map or deepspeed) and not awq"

test_cli_cuda_pytorch_single_gpu:
	pytest -s -k "cli and cuda and pytorch and not (dp or ddp or device_map or deepspeed) and not awq"

test_cli_cuda_torch_ort_multi_gpu:
	pytest -s -k "cli and cuda and torch-ort and (dp or ddp or device_map or deepspeed) and not peft"

test_cli_cuda_torch_ort_single_gpu:
	pytest -s -k "cli and cuda and torch-ort and not (dp or ddp or device_map or deepspeed) and not peft"

test_cli_rocm_pytorch_multi_gpu:
	pytest -s -k "cli and cuda and pytorch and (dp or ddp or device_map or deepspeed) and not bnb"

test_cli_rocm_pytorch_single_gpu:
	pytest -s -k "cli and cuda and pytorch and not (dp or ddp or device_map or deepspeed) and not bnb"

test_cli_llama_cpp:
	pytest -s -k "llama_cpp"
