# Makefile for Optimum-Benchmark

PWD := $(shell pwd)
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

# All targets are phony (don't create files)
.PHONY: help setup install install-dev clean lock update \
	quality style check-format check-lint format lint-fix \
	test test-verbose test-coverage \
	test-api-cpu test-api-cuda test-api-rocm \
	test-cli-cpu-pytorch test-cli-cpu-openvino test-cli-cpu-onnxruntime test-cli-cpu-ipex test-cli-cpu-llama-cpp \
	test-cli-cuda-pytorch-single test-cli-cuda-pytorch-multi \
	test-cli-cuda-onnxruntime test-cli-cuda-tensorrt-llm \
	test-cli-cuda-vllm-single test-cli-cuda-vllm-multi \
	test-cli-rocm-pytorch-single test-cli-rocm-pytorch-multi \
	test-cli-mps-pytorch \
	build-cpu-image build-cuda-image build-rocm-image \
	run-cpu-container run-cuda-container run-rocm-container \
	run-trt-llm-container run-vllm-container

# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "🚀 Development Setup:"
	@echo "  setup               - Set up the project for development"
	@echo "  install             - Install basic dependencies"
	@echo "  install-dev         - Install development dependencies"
	@echo ""
	@echo "🎯 Backend-Specific Installation:"
	@echo "  install-cpu-pytorch        - Install for CPU PyTorch backend"
	@echo "  install-cpu-openvino       - Install for CPU OpenVINO backend"
	@echo "  install-cpu-onnxruntime    - Install for CPU ONNXRuntime backend"
	@echo "  install-cpu-ipex           - Install for CPU Intel Extension for PyTorch"
	@echo "  install-cpu-llama-cpp      - Install for CPU LLaMA-CPP backend"
	@echo "  install-cuda-pytorch       - Install for CUDA PyTorch backend"
	@echo "  install-cuda-onnxruntime   - Install for CUDA ONNXRuntime backend"
	@echo "  install-cuda-tensorrt-llm  - Install for CUDA TensorRT-LLM backend"
	@echo "  install-cuda-vllm          - Install for CUDA vLLM backend"
	@echo "  install-rocm-pytorch       - Install for ROCm PyTorch backend"
	@echo "  install-mps-pytorch        - Install for MPS (Apple Silicon) PyTorch backend"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  quality             - Run quality checks (linting and formatting)"
	@echo "  style               - Fix code formatting and linting issues"
	@echo "  check-format        - Check code formatting only"
	@echo "  check-lint          - Check linting only"
	@echo "  format              - Fix code formatting only"
	@echo "  lint-fix            - Fix linting issues only"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test                - Run basic tests"
	@echo "  test-verbose        - Run tests with verbose output"
	@echo "  test-coverage       - Run tests with coverage report"
	@echo ""
	@echo "🎯 Backend-Specific Testing:"
	@echo "  test-api-cpu               - Test API for CPU backend"
	@echo "  test-api-cuda              - Test API for CUDA backend"
	@echo "  test-api-rocm              - Test API for ROCm backend"
	@echo "  test-cli-cpu-pytorch           - Test CPU PyTorch backend"
	@echo "  test-cli-cpu-openvino          - Test CPU OpenVINO backend"
	@echo "  test-cli-cpu-onnxruntime       - Test CPU ONNXRuntime backend"
	@echo "  test-cli-cpu-ipex              - Test CPU Intel Extension for PyTorch"
	@echo "  test-cli-cpu-llama-cpp         - Test CPU LLaMA-CPP backend"
	@echo "  test-cli-cuda-pytorch-single   - Test CUDA PyTorch (single GPU)"
	@echo "  test-cli-cuda-pytorch-multi    - Test CUDA PyTorch (multi GPU)"
	@echo "  test-cli-cuda-onnxruntime      - Test CUDA ONNXRuntime backend"
	@echo "  test-cli-cuda-tensorrt-llm     - Test CUDA TensorRT-LLM backend"
	@echo "  test-cli-cuda-vllm-single      - Test CUDA vLLM (single GPU)"
	@echo "  test-cli-cuda-vllm-multi       - Test CUDA vLLM (multi GPU)"
	@echo "  test-cli-rocm-pytorch-single   - Test ROCm PyTorch backend (single GPU)"
	@echo "  test-cli-rocm-pytorch-multi    - Test ROCm PyTorch backend (multi GPU)"
	@echo "  test-cli-mps-pytorch           - Test MPS PyTorch backend"
	@echo ""
	@echo "📦 Dependencies:"
	@echo "  lock                - Update lock file"
	@echo "  update              - Update dependencies"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean               - Clean up build artifacts"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  build-cpu-image        - Build CPU Docker image"
	@echo "  build-cuda-image       - Build CUDA Docker image"
	@echo "  build-rocm-image       - Build ROCm Docker image"
	@echo "  run-cpu-container      - Run CPU Docker container"
	@echo "  run-cuda-container     - Run CUDA Docker container"
	@echo "  run-rocm-container     - Run ROCm Docker container"
	@echo "  run-trt-llm-container  - Run TensorRT-LLM Docker container"
	@echo "  run-vllm-container     - Run vLLM Docker container"

# Development setup
setup:
	@echo "Setting up optimum-benchmark for development..."
	@echo "Note: This installs basic dependencies. Use 'make install-<backend>' for specific backends."
	uv sync --dev
	@echo "✅ Project setup complete!"

# Code quality
quality:
	@echo "Running quality checks..."
	uv run ruff format --check .
	uv run ruff check .
	@echo "✅ All quality checks passed!"

style:
	@echo "Fixing code issues..."
	uv run ruff format .
	uv run ruff check --fix .
	@echo "✅ Code fixes applied!"

# Backend-specific installations
install-cpu-pytorch:
	uv sync

install-cpu-openvino:
	uv sync --extra openvino

install-cpu-onnxruntime:
	uv sync --extra onnxruntime

install-cpu-llama-cpp:
	uv sync --extra llama-cpp

install-cpu-ipex:
	uv sync --extra ipex

install-mps-pytorch:
	uv sync

install-rocm-pytorch:
	uv sync

install-cuda-pytorch:
	uv sync --extra bitsandbytes --extra deepspeed

install-cuda-onnxruntime:
	uv sync --extra onnxruntime-gpu

install-cuda-tensorrt-llm:
	uv sync --extra tensorrt-llm

install-cuda-vllm:
	uv sync --extra vllm

lock:
	uv lock

update:
	@echo "Updating dependencies..."
	uv lock --upgrade
	uv sync
	@echo "✅ Dependencies updated!"

# Clean up
clean:
	@echo "Cleaning up build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf sweeps/
	rm -rf runs/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf .pytype/
	rm -rf .venv/
	rm -rf external_repos/
	rm -rf **/__pycache__/
	rm -rf **/*.pyc
	rm -f *.json
	@echo "✅ Cleanup complete!"

# Testing
## API tests
test-api-cpu:
	uv sync --dev
	uv run pytest tests/test_api.py -s -k "api and cpu"

test-api-cuda:
	uv sync --dev
	uv run pytest tests/test_api.py -s -k "api and cuda"

test-api-rocm:
	uv sync --dev
	uv run pytest tests/test_api.py -s -k "api and rocm"

test-api-misc:
	uv sync --dev
	uv run pytest tests/test_api.py -s -k "api and not (cpu or cuda or rocm or mps)"

## API examples
test-api-cpu-examples:
	uv sync --dev
	uv run pytest tests/test_examples.py -s -k "api and cpu and pytorch"

test-api-cuda-examples:
	uv sync --dev --extra torchao
	uv run pytest tests/test_examples.py -s -k "api and cuda and pytorch"

test-api-rocm-examples:
	uv sync --dev --extra torchao
	uv run pytest tests/test_examples.py -s -k "api and rocm and pytorch"

## CLI tests
### CPU tests
test-cli-cpu-pytorch:
	uv sync --dev
	uv run pytest -s -k "cli and cpu and pytorch"

test-cli-cpu-openvino:
	uv sync --dev --extra openvino
	uv run pytest -s -k "cli and cpu and openvino"

test-cli-cpu-onnxruntime:
	uv sync --dev --extra onnxruntime
	uv run pytest -s -k "cli and cpu and onnxruntime"

test-cpu-ipex:
	uv sync --dev --extra ipex
	uv run pytest -s -k "cli and cpu and ipex"

test-cpu-llama-cpp:
	uv sync --dev --extra llama-cpp
	uv run pytest -s -k "llama_cpp"

### CPU examples
test-cli-cpu-pytorch-examples:
	uv sync --dev
	uv run pytest tests/test_examples.py -s -k "cli and cpu and pytorch"

test-cli-cpu-openvino-examples:
	uv sync --dev --extra openvino
	uv run pytest tests/test_examples.py -s -k "cli and cpu and openvino"

test-cli-cpu-onnxruntime-examples:
	uv sync --dev --extra onnxruntime
	uv run pytest tests/test_examples.py -s -k "cli and cpu and onnxruntime"

test-cli-cpu-ipex-examples:
	uv sync --dev --extra ipex
	uv run pytest tests/test_examples.py -s -k "cli and cpu and ipex"

test-cli-cpu-llama-cpp-examples:
	uv sync --dev --extra llama-cpp
	uv run pytest tests/test_examples.py -s -k "cli and cpu and llama-cpp"

### CUDA tests
test-cli-cuda-pytorch-single:
	uv sync --dev --extra bitsandbytes --extra deepspeed
	uv run pytest -s -k "cli and cuda and pytorch and not (dp or ddp or device_map or deepspeed)"

test-cli-cuda-pytorch-multi:
	uv sync --dev --extra bitsandbytes --extra deepspeed
	uv run pytest -s -k "cli and cuda and pytorch and (dp or ddp or device_map or deepspeed)"

test-cli-cuda-vllm-single:
	uv sync --dev --extra vllm
	FORCE_SEQUENTIAL=1 uv run pytest tests/test_cli.py -s -k "cli and cuda and vllm and not (tp or pp)"

test-cli-cuda-vllm-multi:
	uv sync --dev --extra vllm
	FORCE_SEQUENTIAL=1 uv run pytest tests/test_cli.py -s -k "cli and cuda and vllm and (tp or pp)"

test-cli-cuda-tensorrt-llm-single:
	uv sync --dev --extra tensorrt-llm
	FORCE_SEQUENTIAL=1 uv run pytest tests/test_cli.py -s -k "cli and cuda and tensorrt_llm and not (tp or pp)"

test-cli-cuda-tensorrt-llm-multi:
	uv sync --dev --extra tensorrt-llm
	FORCE_SEQUENTIAL=1 uv run pytest tests/test_cli.py -s -k "cli and cuda and tensorrt_llm and (tp or pp)"

test-cli-cuda-onnxruntime:
	uv sync --dev --extra onnxruntime-gpu
	uv run pytest -s -k "cli and cuda and onnxruntime"

### CUDA examples
test-cli-cuda-pytorch-single-examples:
	uv sync --dev --extra bitsandbytes --extra deepspeed
	uv run pytest tests/test_examples.py -s -k "cli and cuda and pytorch and not (dp or ddp or device_map or deepspeed)"

test-cli-cuda-pytorch-multi-examples:
	uv sync --dev --extra bitsandbytes --extra deepspeed
	uv run pytest tests/test_examples.py -s -k "cli and cuda and pytorch and (dp or ddp or device_map or deepspeed)"

test-cli-cuda-onnxruntime-examples:
	uv sync --dev --extra onnxruntime-gpu
	uv run pytest tests/test_examples.py -s -k "cli and cuda and onnxruntime"

test-cli-cuda-vllm-single-examples:
	uv sync --dev --extra vllm
	FORCE_SEQUENTIAL=1 uv run pytest tests/test_examples.py -s -k "cli and cuda and vllm and not (tp or pp)"

test-cli-cuda-vllm-multi-examples:
	uv sync --dev --extra vllm
	FORCE_SEQUENTIAL=1 uv run pytest tests/test_examples.py -s -k "cli and cuda and vllm and (tp or pp)"

test-cli-cuda-tensorrt-llm-single-examples:
	uv sync --dev --extra tensorrt-llm
	FORCE_SEQUENTIAL=1 uv run pytest tests/test_examples.py -s -k "cli and cuda and tensorrt_llm and not (tp or pp)"

test-cli-cuda-tensorrt-llm-multi-examples:
	uv sync --dev --extra tensorrt-llm
	FORCE_SEQUENTIAL=1 uv run pytest tests/test_examples.py -s -k "cli and cuda and tensorrt_llm and (tp or pp)"

### ROCm tests
test-cli-rocm-pytorch-single:
	uv sync --dev
	uv run pytest -s -k "cli and cuda and pytorch and not (dp or ddp or device_map or deepspeed)"

test-cli-rocm-pytorch-multi:
	uv sync --dev
	uv run pytest -s -k "cli and cuda and pytorch and (dp or ddp or device_map or deepspeed)"

test-cli-rocm-pytorch-examples:
	uv sync --dev
	uv run pytest tests/test_examples.py -s -k "cli and rocm and pytorch"

### MPS tests
test-cli-mps-pytorch:
	uv sync --dev
	uv run pytest -s -k "cli and mps and pytorch"

test-cli-mps-pytorch-examples:
	uv sync --dev
	uv run pytest tests/test_examples.py -s -k "cli and mps and pytorch"

### MISC CLI tests
test-cli-misc:
	uv sync --dev
	uv run pytest tests/test_cli.py -s -k "cli and not (cpu or cuda or rocm or mps)"

### Energy Star tests/examples
test-energy-star:
	uv sync --dev
	uv run pytest tests/test_energy_star.py -vvvv

# Build docker
build-cpu-image:
	docker build -t optimum-benchmark:latest-cpu -f docker/cpu/Dockerfile .
	docker build --build-arg IMAGE=optimum-benchmark:latest-cpu --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-cpu docker/unroot

build-cuda-image:
	docker build -t optimum-benchmark:latest-cuda -f docker/cuda/Dockerfile .
	docker build --build-arg IMAGE=optimum-benchmark:latest-cuda --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-cuda docker/unroot

build-rocm-image:
	docker build -t optimum-benchmark:latest-rocm -f docker/rocm/Dockerfile .
	docker build --build-arg IMAGE=optimum-benchmark:latest-rocm --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t optimum-benchmark:latest-rocm docker/unroot

# Run docker
run-cpu-container:
	docker run \
	-it \
	--rm \
	--ipc host \
	--pid host \
	--shm-size 64G \
	--volume $(PWD):/optimum-benchmark \
	--workdir /optimum-benchmark \
	ghcr.io/huggingface/optimum-benchmark:latest-cpu

run-cuda-container:
	docker run \
	-it \
	--rm \
	--ipc host \
	--pid host \
	--gpus all \
	--shm-size 64G \
	--volume $(PWD):/optimum-benchmark \
	--workdir /optimum-benchmark \
	ghcr.io/huggingface/optimum-benchmark:latest-cuda

run-rocm-container:
	docker run \
	-it \
	--rm \
	--ipc host \
	--pid host \
	--shm-size 64G \
	--device /dev/kfd \
	--device /dev/dri \
	--group-add video \
	--volume $(PWD):/optimum-benchmark \
	--workdir /optimum-benchmark \
	ghcr.io/huggingface/optimum-benchmark:latest-rocm

run-trt-llm-container:
	docker run \
	-it \
	--rm \
	--ipc host \
	--pid host \
	--gpus all \
	--shm-size 64G \
	--volume $(PWD):/optimum-benchmark \
	--workdir /optimum-benchmark \
	huggingface/optimum-nvidia:latest

run-vllm-container:
	docker run \
	-it \
	--rm \
	--ipc host \
	--pid host \
	--gpus all \
	--shm-size 64G \
	--volume $(PWD):/optimum-benchmark \
	--workdir /optimum-benchmark \
	--entrypoint /bin/bash \
	vllm/vllm-openai:latest

