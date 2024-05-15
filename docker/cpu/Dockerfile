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
ENV DEBIAN_FRONTEND noninteractive
ENV PATH="/home/user/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libglib2.0-0 \
    sudo build-essential git bash-completion \
    python3.10 python3-pip python3.10-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    pip install --no-cache-dir --upgrade pip setuptools wheel 

# Install PyTorch
ARG TORCH_VERSION=stable

RUN if [ "${TORCH_VERSION}" = "stable" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ; \
elif [ "${TORCH_VERSION}" = "nighly" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu ; \
else \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ; \
fi
