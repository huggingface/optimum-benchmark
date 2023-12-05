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

ARG ROCM_VERSION=5.7
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10

FROM rocm/pytorch:rocm${ROCM_VERSION}_ubuntu${UBUNTU_VERSION}_py${PYTHON_VERSION}_pytorch_2.0.1

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Run as non-root user
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# Install and update tools to minimize security vulnerabilities - are all of these really necessary?
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    apt-utils \
    patchelf \
    git \
    libprotobuf-dev \
    protobuf-compiler \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    mercurial \
    subversion \
    libopenmpi-dev \
    ffmpeg && \
    apt-get clean && \
    apt-get autoremove -y

# Update pip
RUN pip install --upgrade pip
# Install dependencies
RUN pip install cmake onnx ninja transformers --no-cache-dir
# Install ONNXRuntime from source
RUN git clone --recursive https://github.com/ROCmSoftwarePlatform/onnxruntime.git && cd onnxruntime && git checkout rocm5.7_internal_testing_eigen-3.4.zip_hash
RUN cd onnxruntime && ./build.sh --config Release --build_wheel --allow_running_as_root --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) --use_rocm --rocm_home=/opt/rocm
RUN pip install onnxruntime/build/Linux/Release/dist/*
RUN pip install git+https://github.com/huggingface/optimum.git

# amd-smi must be installed before switching to user
RUN apt-get update && apt-get upgrade -y && apt-get -y --no-install-recommends install amd-smi-lib
RUN pip install --upgrade pip setuptools wheel && cd /opt/rocm/share/amd_smi && pip install .
ENV PATH="/opt/rocm/bin:${PATH}"

# Add local bin to PATH
ENV PATH="/home/user/.local/bin:${PATH}"

# Add user to sudoers
RUN adduser user sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >>/etc/sudoers

# Fix AMD permissions
RUN usermod -g video user
RUN usermod -a -G render user

# Change user
USER user
WORKDIR /home/user
