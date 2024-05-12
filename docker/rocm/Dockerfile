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

ARG ROCM_VERSION=5.7.1
ARG UBUNTU_VERSION=22.04

FROM rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}

# Install necessary packages
ENV PATH="/opt/rocm/bin:${PATH}"
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \ 
    sudo build-essential git bash-completion \
    python3.10 python3-pip python3.10-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    pip install --no-cache-dir --upgrade pip setuptools wheel  && \
    cd /opt/rocm/share/amd_smi && pip install .

# Install PyTorch
ARG TORCH_ROCM=rocm5.7
ARG TORCH_VERSION=2.2.2

RUN if [ "${TORCH_VERSION}" = "stable" ]; then \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_ROCM} ; \
elif [ "${TORCH_VERSION}" = "nighly" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_ROCM} ; \
else \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_ROCM} ; \
fi
