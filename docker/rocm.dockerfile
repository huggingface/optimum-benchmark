FROM rocm/dev-ubuntu-22.04:5.6.1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install -U pip

ENV PATH="/home/user/.local/bin:${PATH}"

# We need ONNX here because the dummy input generator relies on the ONNX config in Optimum, which is unwanted and needs to be fixed.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
RUN pip install --no-cache-dir transformers accelerate optimum omegaconf hydra-core hydra_colorlog psutil pandas onnx

RUN git clone https://github.com/RadeonOpenCompute/pyrsmi.git && cd pyrsmi && pip install -e .