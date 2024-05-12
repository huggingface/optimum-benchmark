# List of targets that are not associated with files
.PHONY: quality style install build_cpu_image build_cuda_118_image build_cuda_121_image build_rocm_image run_cpu_container run_cuda_118_container run_cuda_121_container run_rocm_container install_api_misc install_api_cpu install_api_cuda install_api_rocm install_cli_misc install_cli_cpu_pytorch install_cli_cpu_openvino install_cli_cpu_onnxruntime install_cli_cpu_neural_compressor install_cli_cuda_pytorch install_cli_rocm_pytorch install_cli_cuda_torch_ort install_cli_cuda_onnxruntime test_api_misc test_api_cpu test_api_cuda test_api_rocm test_cli_misc test_cli_cpu_pytorch test_cli_cpu_openvino test_cli_cpu_onnxruntime test_cli_cpu_neural_compressor test_cli_cuda_onnxruntime test_cli_cuda_pytorch_multi_gpu test_cli_cuda_pytorch_single_gpu test_cli_cuda_torch_ort_multi_gpu test_cli_cuda_torch_ort_single_gpu test_cli_rocm_pytorch_multi_gpu test_cli_rocm_pytorch_single_gpu install_llm_perf_cuda_pytorch run_llm_perf_cuda_pytorch_unquantized run_llm_perf_cuda_pytorch_bnb run_llm_perf_cuda_pytorch_gptq run_llm_perf_cuda_pytorch_awq

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
	docker build -t optimum-benchmark:latest-cpu docker/cpu
	docker build --build-arg IMAGE=optimum-benchmark:latest-cpu --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-cpu docker/unroot

build_cuda_image:
	docker build --build-arg -t optimum-benchmark:latest-cuda-ort docker/cuda-ort
	docker build --build-arg IMAGE=optimum-benchmark:latest-cuda-ort --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-cuda-ort docker/unroot

build_cuda_ort_image:
	docker build --build-arg -t optimum-benchmark:latest-cuda docker/cuda
	docker build --build-arg IMAGE=optimum-benchmark:latest-cuda --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-cuda docker/unroot

build_rocm_image:
	docker build --build-arg -t optimum-benchmark:latest-rocm docker/rocm
	docker build --build-arg IMAGE=optimum-benchmark:latest-rocm --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-rocm docker/unroot

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
	pip install -e .[testing,timm,diffusers,peft,autoawq,auto-gptq,bitsandbytes,deepspeed]

install_cli_rocm_pytorch:
	pip install -e .[testing,timm,diffusers,peft,autoawq,auto-gptq,deepspeed]

install_cli_cuda_torch_ort:
	pip install -e .[testing,timm,diffusers,peft,torch-ort,deepspeed]

install_cli_cuda_onnxruntime:
	pip install -e .[testing,timm,diffusers,peft,onnxruntime-gpu]

# Run tests

test_api_misc:
	pytest -s -k "api and not (cpu or cuda)

test_api_cpu:
	pytest -s -k "api and cpu"

test_api_cuda:
	pytest -s -k "api and cuda"

test_api_rocm:
	pytest -s -k "api and cuda"

test_cli_misc:
	pytest -s -k "cli and not (cpu or cuda)"

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

test_cli_cuda_pytorch_multi_gpu:
	pytest -s -k "cli and cuda and pytorch and (dp or ddp or device_map or deepspeed) and not awq"

test_cli_cuda_pytorch_single_gpu:
	pytest -s -k "cli and cuda and pytorch and not (dp or ddp or device_map or deepspeed) and not awq"

test_cli_cuda_torch_ort_multi_gpu:
	pytest -s -k "cli and cuda and torch-ort and (dp or ddp or device_map or deepspeed) and not peft"

test_cli_cuda_torch_ort_single_gpu:
	pytest -s -k "cli and cuda and torch-ort and not (dp or ddp or device_map or deepspeed) and not peft"

test_cli_rocm_pytorch_multi_gpu:
	pytest -s -k "cli and rocm and pytorch and (dp or ddp or device_map or deepspeed) and not (bnb or awq)"

test_cli_rocm_pytorch_single_gpu:
	pytest -s -k "cli and rocm and pytorch and not (dp or ddp or device_map or deepspeed) and not (bnb or awq)"

# llm-perf

install_llm_perf_cuda_pytorch:
	pip install packaging && pip install flash-attn einops scipy auto-gptq optimum bitsandbytes autoawq codecarbon
	pip install -U transformers huggingface_hub[hf_transfer]
	pip install -e .

run_llm_perf_cuda_pytorch_unquantized:
	SUBSET=unquantized python llm_perf/benchmark_cuda_pytorch.py

run_llm_perf_cuda_pytorch_bnb:
	SUBSET=bnb python llm_perf/benchmark_cuda_pytorch.py

run_llm_perf_cuda_pytorch_gptq:
	SUBSET=gptq python llm_perf/benchmark_cuda_pytorch.py

run_llm_perf_cuda_pytorch_awq:
	SUBSET=awq python llm_perf/benchmark_cuda_pytorch.py
