# Makefile for Optimum-Benchmark

PWD := $(shell pwd)
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

# All targets are phony (don't create files)
.PHONY: help setup install install-dev \
	install-pytorch install-ipex install-vllm install-openvino install-llama-cpp install-onnxruntime install-onnxruntime-gpu \
	lock update clean quality style check-format check-lint format lint-fix \
	test test-verbose test-coverage \
	test-api-cpu test-api-cuda test-api-rocm test-api-misc \
	test-api-cpu-examples test-api-cuda-examples test-api-rocm-examples \
	test-cli-cpu-pytorch test-cli-cpu-openvino test-cli-cpu-py-txi test-cli-cpu-onnxruntime test-cli-cpu-ipex test-cli-cpu-llama-cpp \
	test-cli-cpu-pytorch-examples test-cli-cpu-openvino-examples test-cli-cpu-onnxruntime-examples test-cli-cpu-py-txi-examples test-cli-cpu-llama-cpp-examples \
	test-cli-cuda-pytorch-single test-cli-cuda-pytorch-multi \
	test-cli-cuda-vllm-single test-cli-cuda-vllm-multi \
	test-cli-cuda-tensorrt-llm-single test-cli-cuda-tensorrt-llm-multi \
	test-cli-cuda-onnxruntime test-cli-cuda-py-txi \
	test-cli-rocm-pytorch-single test-cli-rocm-pytorch-multi \
	test-cli-rocm-pytorch-single-examples test-cli-rocm-pytorch-multi-examples \
	test-cli-mps-pytorch test-cli-mps-pytorch-examples \
	test-energy-star build-cpu-image build-cuda-image build-rocm-image run-cpu-container run-cuda-container run-rocm-container run-tensorrt-llm-container run-vllm-container

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
	@echo "  install-pytorch         - Install CPU PyTorch backend"
	@echo "  install-ipex            - Install CPU IPEX backend"
	@echo "  install-vllm            - Install vLLM backend"
	@echo "  install-openvino        - Install CPU OpenVINO backend"
	@echo "  install-llama-cpp       - Install CPU LLaMA-CPP backend"
	@echo "  install-onnxruntime     - Install CPU ONNXRuntime backend"
	@echo "  install-onnxruntime-gpu - Install GPU ONNXRuntime backend"
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
	@echo "  test-api-cpu                          - Test API for CPU backend"
	@echo "  test-api-cuda                         - Test API for CUDA backend"
	@echo "  test-api-rocm                         - Test API for ROCm backend"
	@echo "  test-api-misc                         - Test API for miscellaneous backends"
	@echo "  test-api-cpu-examples                 - Test API examples for CPU backend"
	@echo "  test-api-cuda-examples                - Test API examples for CUDA backend"
	@echo "  test-api-rocm-examples                - Test API examples for ROCm backend"
	@echo "  test-cli-cpu-pytorch                  - Test CLI for CPU PyTorch backend"
	@echo "  test-cli-cpu-openvino                 - Test CLI for CPU OpenVINO backend"
	@echo "  test-cli-cpu-py-txi                   - Test CLI for CPU Py-TXI backend"
	@echo "  test-cli-cpu-onnxruntime              - Test CLI for CPU ONNXRuntime backend"
	@echo "  test-cli-cpu-ipex                     - Test CLI for CPU Intel Extension for PyTorch"
	@echo "  test-cli-cpu-llama-cpp                - Test CLI for CPU LLaMA-CPP backend"
	@echo "  test-cli-cpu-pytorch-examples         - Test CLI examples for CPU PyTorch backend"
	@echo "  test-cli-cpu-openvino-examples        - Test CLI examples for CPU OpenVINO backend"
	@echo "  test-cli-cpu-onnxruntime-examples     - Test CLI examples for CPU ONNXRuntime backend"
	@echo "  test-cli-cpu-py-txi-examples          - Test CLI examples for CPU Py-TXI backend"
	@echo "  test-cli-cpu-llama-cpp-examples       - Test CLI examples for CPU LLaMA-CPP backend"
	@echo "  test-cli-cuda-pytorch-single          - Test CLI for single GPU CUDA PyTorch backend"
	@echo "  test-cli-cuda-pytorch-multi           - Test CLI for multi GPU CUDA PyTorch backend"
	@echo "  test-cli-cuda-vllm-single             - Test CLI for single GPU CUDA vLLM backend"
	@echo "  test-cli-cuda-vllm-multi              - Test CLI for multi GPU CUDA vLLM backend"
	@echo "  test-cli-cuda-tensorrt-llm-single     - Test CLI for single GPU CUDA TensorRT-LLM backend"
	@echo "  test-cli-cuda-tensorrt-llm-multi      - Test CLI for multi GPU CUDA TensorRT-LLM backend"
	@echo "  test-cli-cuda-onnxruntime             - Test CLI for CUDA ONNXRuntime backend"
	@echo "  test-cli-cuda-py-txi                  - Test CLI for CUDA Py-TXI backend"
	@echo "  test-cli-rocm-pytorch-single          - Test CLI for single GPU ROCm PyTorch backend"
	@echo "  test-cli-rocm-pytorch-multi           - Test CLI for multi GPU ROCm PyTorch backend"
	@echo "  test-cli-rocm-pytorch-single-examples - Test CLI examples for single GPU ROCm PyTorch backend"
	@echo "  test-cli-rocm-pytorch-multi-examples  - Test CLI examples for multi GPU ROCm PyTorch backend"
	@echo "  test-cli-mps-pytorch                  - Test CLI for MPS (Apple Silicon) PyTorch backend"
	@echo "  test-cli-mps-pytorch-examples         - Test CLI examples for MPS (Apple Silicon) PyTorch backend"
	@echo "  test-cli-misc                         - Test CLI for miscellaneous backends"
	@echo "  test-energy-star                      - Run Energy Star tests"
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
	@echo "  run-tensorrt-llm-container  - Run TensorRT-LLM Docker container"
	@echo "  run-vllm-container     - Run vLLM Docker container"

# Development setup
setup:
	@echo "Setting up optimum-benchmark for development..."
	@echo "Note: This installs basic dependencies. Use 'make install-<backend>' for specific backends."
	uv sync --dev
	@echo "✅ Project setup complete!"

# Run quality checks
quality:
	@echo "Running quality checks..."
	uv run ruff format --check .
	uv run ruff check .
	@echo "✅ All quality checks passed!"

# Apply code style fixes
style:
	@echo "Fixing code issues..."
	uv run ruff format .
	uv run ruff check --fix .
	@echo "✅ Code fixes applied!"

# Install dependencies and update the lock file
update:
	@echo "Updating dependencies..."
	uv lock --upgrade
	uv sync
	@echo "✅ Dependencies updated!"

# Build the project and create wheels
build:
	@echo "Building the project..."
	uv build
	@echo "✅ Build complete!"

# Release the project to PyPI
release:
	@echo "Releasing the project to PyPI..."
	uv build
	uv run twine upload dist/*
	@echo "✅ Release complete!"

# Clean up
clean:
	@echo "Cleaning up build artifacts..."
	rm -rf dist/
	rm -rf runs/
	rm -rf build/
	rm -rf sweeps/
	rm -rf outputs/
	rm -rf external_repos/
	rm -rf trainer_output/
	rm -rf optimum_benchmark.egg-info/
	rm -rf .venv/
	rm -rf .pytype/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf **/__pycache__/
	rm -rf **/*.pyc
	rm -rf *.json
	rm -rf *.log
	@echo "✅ Cleanup complete!"

# Backend-specific installations
install-pytorch:
	uv sync

install-ipex:
	uv sync --extra ipex

install-vllm:
	uv sync --extra vllm

install-openvino:
	uv sync --extra openvino

install-llama-cpp:
	uv sync --extra llama-cpp

install-onnxruntime:
	uv sync --extra onnxruntime

install-onnxruntime-gpu:
	uv sync --extra onnxruntime-gpu

install-tensorrt-llm:
	UV_SYSTEM_PYTHON=1 uv pip install -e .[dev,tensorrt-llm]

# Testing
## API tests
test-api-cpu:
	uv run pytest tests/test_api.py -s -v -k "api and cpu"

test-api-cuda:
	uv run pytest tests/test_api.py -s -v -k "api and cuda"

test-api-rocm:
	uv run pytest tests/test_api.py -s -v -k "api and rocm"

test-api-misc:
	uv run pytest tests/test_api.py -s -v -k "api and not (cpu or cuda or rocm or mps)"

## API examples
test-api-cpu-examples:
	uv run pytest tests/test_examples.py -s -v -k "api and cpu "

test-api-cuda-examples:
	uv run --extra torchao pytest tests/test_examples.py -s -v -k "api and cuda"

test-api-rocm-examples:
	uv run --extra torchao pytest tests/test_examples.py -s -v -k "api and rocm"

## CLI tests
### CPU tests
test-cli-cpu-pytorch:
	uv run pytest tests/test_cli.py -s -v -k "cli and cpu and pytorch"

test-cli-cpu-openvino:
	uv run --extra openvino pytest tests/test_cli.py -s -v -k "cli and cpu and openvino"

test-cli-cpu-py-txi:
	uv run --extra py-txi pytest tests/test_cli.py -s -v -k "cli and cpu and (tgi or tei or txi)"

test-cli-cpu-onnxruntime:
	uv run --extra onnxruntime pytest tests/test_cli.py -s -v -k "cli and cpu and onnxruntime"

test-cli-cpu-ipex:
	uv run --extra ipex pytest tests/test_cli.py -s -v -k "cli and cpu and ipex"

test-cli-cpu-llama-cpp:
	uv run --extra llama-cpp pytest tests/test_cli.py -s -v -k "llama_cpp"

### CPU examples
test-cli-cpu-pytorch-examples:
	uv run pytest tests/test_examples.py -s -v -k "cli and cpu and pytorch"

test-cli-cpu-openvino-examples:
	uv run --extra openvino pytest tests/test_examples.py -s -v -k "cli and cpu and openvino"

test-cli-cpu-onnxruntime-examples:
	uv run --extra onnxruntime pytest tests/test_examples.py -s -v -k "cli and cpu and onnxruntime"

test-cli-cpu-py-txi-examples:
	uv run --extra py-txi pytest tests/test_examples.py -s -v -k "cli and cpu and (tgi or tei or txi)"

test-cli-cpu-llama-cpp-examples:
	uv run --extra llama-cpp pytest tests/test_examples.py -s -v -k "cli and cpu and llama-cpp"

test-cli-cpu-ipex-examples:
	uv run --extra ipex pytest tests/test_examples.py -s -v -k "cli and cpu and ipex"

### CUDA tests
test-cli-cuda-onnxruntime:
	uv run --extra onnxruntime-gpu pytest tests/test_cli.py -s -v -k "cli and cuda and onnxruntime"

test-cli-cuda-pytorch-single:
	uv run pytest tests/test_cli.py -s -v -k "cli and cuda and pytorch and not (tp or dp or ddp or device_map or deepspeed) and not (bnb or gptq)"

test-cli-cuda-pytorch-multi:
	FORCE_SEQUENTIAL=1 uv run --extra deepspeed pytest tests/test_cli.py -s -v -k "cli and cuda and pytorch and (tp or dp or ddp or device_map or deepspeed)"

test-cli-cuda-vllm-single:
	FORCE_SEQUENTIAL=1 uv run --extra vllm pytest tests/test_cli.py -s -v -k "cli and cuda and vllm and not (tp or pp)"

test-cli-cuda-vllm-multi:
	FORCE_SEQUENTIAL=1 uv run --extra vllm pytest tests/test_cli.py -s -v -k "cli and cuda and vllm and (tp or pp)"

test-cli-cuda-py-txi:
	FORCE_SEQUENTIAL=1 uv run --extra py-txi pytest tests/test_cli.py -s -v -k "cli and cuda and (tgi or tei or txi)"

#### non-uv compatible
test-cli-cuda-tensorrt-llm-single:
	UV_SYSTEM_PYTHON=1 uv pip install -e .[dev,tensorrt-llm]
	FORCE_SEQUENTIAL=1 pytest tests/test_cli.py -s -v -k "cli and cuda and tensorrt_llm and not (tp or pp)"

test-cli-cuda-tensorrt-llm-multi:
	UV_SYSTEM_PYTHON=1 uv pip install -e .[dev,tensorrt-llm]
	FORCE_SEQUENTIAL=1 pytest tests/test_cli.py -s -v -k "cli and cuda and tensorrt_llm and (tp or pp)"

### CUDA examples
test-cli-cuda-onnxruntime-examples:
	uv run --extra onnxruntime-gpu pytest tests/test_examples.py -s -v -k "cli and cuda and onnxruntime"

test-cli-cuda-pytorch-single-examples:
	uv run pytest tests/test_examples.py -s -v -k "cli and cuda and pytorch and not (tp or dp or ddp or device_map or deepspeed)"

test-cli-cuda-pytorch-multi-examples:
	FORCE_SEQUENTIAL=1 uv run --extra deepspeed pytest tests/test_examples.py -s -v -k "cli and cuda and pytorch and (tp or dp or ddp or device_map or deepspeed)"

test-cli-cuda-vllm-single-examples:
	FORCE_SEQUENTIAL=1 uv run --extra vllm pytest tests/test_examples.py -s -v -k "cli and cuda and vllm and not (tp or pp)"

test-cli-cuda-vllm-multi-examples:
	FORCE_SEQUENTIAL=1 uv run --extra vllm pytest tests/test_examples.py -s -v -k "cli and cuda and vllm and (tp or pp)"

test-cli-cuda-py-txi-examples:
	FORCE_SEQUENTIAL=1 uv run --extra py-txi pytest tests/test_examples.py -s -v -k "cli and cuda and (tgi or tei or txi)"

#### non-uv compatible
test-cli-cuda-tensorrt-llm-single-examples:
	pip install uv --upgrade
	UV_SYSTEM_PYTHON=1 uv pip install -e .[dev,tensorrt-llm]
	FORCE_SEQUENTIAL=1 pytest tests/test_examples.py -s -v -k "cli and cuda and tensorrt_llm and not (tp or pp)"

test-cli-cuda-tensorrt-llm-multi-examples:
	pip install uv --upgrade
	UV_SYSTEM_PYTHON=1 uv pip install -e .[dev,tensorrt-llm]
	FORCE_SEQUENTIAL=1 pytest tests/test_examples.py -s -v -k "cli and cuda and tensorrt_llm and (tp or pp)"

### ROCm tests
test-cli-rocm-pytorch-single:
	uv run pytest tests/test_cli.py -s -v -k "cli and cuda and pytorch and not (tp or dp or ddp or device_map or deepspeed)"

test-cli-rocm-pytorch-multi:
	FORCE_SEQUENTIAL=1 uv run pytest tests/test_cli.py -s -v -k "cli and cuda and pytorch and (tp or dp or ddp or device_map or deepspeed)"

### ROCm examples
test-cli-rocm-pytorch-single-examples:
	uv run pytest tests/test_examples.py -s -v -k "cli and rocm and pytorch and not (tp or dp or ddp or device_map or deepspeed)"

test-cli-rocm-pytorch-multi-examples:
	FORCE_SEQUENTIAL=1 uv run pytest tests/test_examples.py -s -v -k "cli and rocm and pytorch and (tp or dp or ddp or device_map or deepspeed)"

### MPS tests
test-cli-mps-pytorch:
	uv run pytest tests/test_cli.py -s -v -k "cli and mps and pytorch"

### MPS examples
test-cli-mps-pytorch-examples:
	uv run pytest tests/test_examples.py -s -v -k "cli and mps and pytorch"

### MISC CLI tests
test-cli-misc:
	uv run pytest tests/test_cli.py -s -v -k "cli and not (cpu or cuda or rocm or mps)"

### Energy Star
test-energy-star:
	uv run pytest tests/test_energy_star.py -s -v

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

run-tensorrt-llm-container:
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
