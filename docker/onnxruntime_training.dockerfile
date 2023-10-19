# Copyright 2023 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# docker build -f docker/onnxruntime_training.dockerfile -t onnxruntime-training .

ARG CUDNN_VERSION=8
ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.9
ARG TORCH_VERSION=2.0.0
ARG TORCHVISION_VERSION=0.15.1

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Run as non-root user
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# Install python
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

# Add user to sudoers
RUN adduser user sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Change user
USER user
WORKDIR /home/user

# Install dependencies
RUN pip install onnx ninja
RUN pip install onnxruntime-training==1.15.1
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}
RUN pip install torch-ort
ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
RUN python3 -m torch_ort.configure
