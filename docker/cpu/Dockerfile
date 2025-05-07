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

ARG UBUNTU_VERSION=22.04

FROM ubuntu:${UBUNTU_VERSION}

# Install necessary packages
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/home/user/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo build-essential git bash-completion numactl \
    python3.10 python3-pip python3.10-dev google-perftools && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    pip install --no-cache-dir --upgrade pip setuptools wheel intel-openmp

ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/local/lib/libiomp5.so"

# Install PyTorch
ARG TORCH_VERSION=""
ARG TORCH_RELEASE_TYPE=stable

RUN if [ -n "${TORCH_VERSION}" ]; then \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/cpu ; \
elif [ "${TORCH_RELEASE_TYPE}" = "stable" ]; then \
    pip install --no-cache-dir torch torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/cpu ; \
elif [ "${TORCH_RELEASE_TYPE}" = "nightly" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/nightly/cpu ; \
else \
    echo "Error: Invalid TORCH_RELEASE_TYPE. Must be 'stable', 'nightly', or specify a TORCH_VERSION." && exit 1 ; \
fi
