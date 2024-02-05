# List of targets that are not associated with files
.PHONY:	style_check style test install install_dev_cpu install_dev_gpu

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .

install:
	pip install -e .

install_dev_cpu:
	pip install -e .[quality,testing,openvino,onnxruntime,neural-compressor,diffusers,timm,peft]

install_dev_gpu:
	pip install -e .[quality,testing,onnxruntime-gpu,deepspeed,diffusers,timm,peft]
