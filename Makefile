# List of targets that are not associated with files
.PHONY:	quality style install build_docker_cpu build_docker_cuda build_docker_rocm build_docker_tensorrt test_api_misc test_api_cpu test_api_cuda test_api_rocm test_api_tensorrt test_cli_misc test_cli_cpu_pytorch test_cli_cpu_neural_compressor test_cli_cpu_onnxruntime test_cli_cpu_openvino test_cli_cuda_pytorch test_cli_rocm_pytorch test_cli_tensorrt_onnxruntime test_cli_tensorrt_llm

, := ,
PWD := $(shell pwd)
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

API_MISC_REQS := testing
API_CPU_REQS := testing,timm,diffusers
API_CUDA_REQS := testing,timm,diffusers
API_ROCM_REQS := testing,timm,diffusers

CLI_MISC_REQS := testing

CLI_CUDA_ONNXRUNTIME_REQS := testing,timm,diffusers
CLI_ROCM_ONNXRUNTIME_REQS := testing,timm,diffusers
CLI_CUDA_PYTORCH_REQS := testing,timm,diffusers,deepspeed,peft
CLI_ROCM_PYTORCH_REQS := testing,timm,diffusers,deepspeed,peft

CLI_CPU_OPENVINO_REQS := testing,openvino,timm,diffusers
CLI_CPU_PYTORCH_REQS := testing,timm,diffusers,deepspeed,peft
CLI_CPU_ONNXRUNTIME_REQS := testing,onnxruntime,timm,diffusers
CLI_CPU_NEURAL_COMPRESSOR_REQS := testing,neural-compressor,timm,diffusers

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .

install:
	pip install -e .

## Docker builds

define build_docker
	docker build -f docker/$(1).dockerfile  --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t opt-bench-$(1):local .
endef

build_docker_cpu:
	$(call build_docker,cpu)

build_docker_cuda:
	$(call build_docker,cuda)

build_docker_rocm:
	$(call build_docker,rocm)

## Tests

define test_ubuntu
	docker run \
	--rm \
	--pid host \
	--entrypoint /bin/bash \
	--volume $(PWD):/workspace \
	--workdir /workspace \
	opt-bench-$(1):local -c "pip install -e .[$(2)] && pytest tests/ -k '$(3)' -x"
endef

define test_nvidia
	docker run \
	--rm \
	--pid host \
	--shm-size 64G \
	--gpus '"device=0,1"' \
	--entrypoint /bin/bash \
	--volume $(PWD):/workspace \
	--workdir /workspace \
	opt-bench-$(1):local -c "pip install -e .[$(2)] && pytest tests/ -k '$(3)' -x"
endef

define test_amdgpu
	docker run \
	--rm \
	--pid host \
	--shm-size 64G \
	--device /dev/kfd \
	--device /dev/dri/renderD128 \
	--device /dev/dri/renderD129 \
	--entrypoint /bin/bash \
	--volume $(PWD):/workspace \
	--workdir /workspace \
	opt-bench-$(1):local -c "pip install -e .[$(2)] && pytest tests/ -k '$(3)' -x"
endef

# group the extra
test_api_cpu:
	$(call test_ubuntu,cpu,$(API_CPU_REQS),api and cpu)

test_api_cuda:
	$(call test_nvidia,cuda,$(API_CUDA_REQS),api and cuda)

test_api_rocm:
	$(call test_amdgpu,rocm,$(API_ROCM_REQS),api and rocm)

test_api_misc:
	$(call test_ubuntu,cpu,$(API_MISC_REQS),api and not (cpu or cuda or rocm or tensorrt))

## CLI tests

test_cli_cuda_pytorch:
	$(call test_nvidia,cuda,$(CLI_CUDA_PYTORCH_REQS),cli and cuda and pytorch)

test_cli_rocm_pytorch:
	$(call test_amdgpu,rocm,$(CLI_ROCM_PYTORCH_REQS),cli and cuda and pytorch and peft)

test_cli_cuda_onnxruntime:
	$(call test_nvidia,cuda,$(CLI_CUDA_ONNXRUNTIME_REQS),cli and cuda and onnxruntime)

test_cli_rocm_onnxruntime:
	$(call test_amdgpu,rocm,$(CLI_ROCM_ONNXRUNTIME_REQS),cli and rocm and onnxruntime)

test_cli_cpu_pytorch:
	$(call test_ubuntu,cpu,$(CLI_CPU_PYTORCH_REQS),cli and cpu and pytorch)

test_cli_cpu_openvino:
	$(call test_ubuntu,cpu,$(CLI_CPU_OPENVINO_REQS),cli and cpu and openvino)

test_cli_cpu_onnxruntime:
	$(call test_ubuntu,cpu,$(CLI_CPU_ONNXRUNTIME_REQS),cli and cpu and onnxruntime)

test_cli_cpu_neural_compressor:
	$(call test_ubuntu,cpu,$(CLI_CPU_NEURAL_COMPRESSOR_REQS),cli and cpu and neural-compressor)
