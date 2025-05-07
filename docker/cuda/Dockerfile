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

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

# Install necessary packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo build-essential git bash-completion \
    python3.10 python3-pip python3.10-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    pip install --no-cache-dir --upgrade pip setuptools wheel requests

# Install PyTorch
ARG TORCH_VERSION=""
ARG TORCH_CUDA=cu128
ARG TORCH_RELEASE_TYPE=stable

RUN if [ -n "${TORCH_VERSION}" ]; then \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/${TORCH_CUDA} ; \
elif [ "${TORCH_RELEASE_TYPE}" = "stable" ]; then \
    pip install --no-cache-dir torch torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/${TORCH_CUDA} ; \
elif [ "${TORCH_RELEASE_TYPE}" = "nightly" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/nightly/${TORCH_CUDA} ; \
else \
    echo "Error: Invalid TORCH_RELEASE_TYPE. Must be 'stable', 'nightly', or specify a TORCH_VERSION." && exit 1 ; \
fi

# Install quantization libraries from source
ENV MAX_JOBS=1
ENV CUDA_VERSION=12.8
ENV GPTQMODEL_FORCE_BUILD=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

RUN pip install -v gptqmodel --no-build-isolation --no-cache-dir
