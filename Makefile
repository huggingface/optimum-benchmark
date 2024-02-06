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

install_cpu_dev:
	pip install -e .[quality,testing,openvino,onnxruntime,neural-compressor,diffusers,timm,peft]

install_gpu_dev:
	pip install -e .[quality,testing,onnxruntime-gpu,deepspeed,diffusers,timm,peft]
