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

ARG TORCH_ROCM=rocm5.7
ARG TORCH_PRE_RELEASE=0

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Run as non-root user
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# Install python
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

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

# Update pip
RUN pip install --upgrade pip

# Install PyTorch (nightly if ROCM_VERSION=5.7 or TORCH_PRE_RELEASE=1)
RUN if [ "${TORCH_PRE_RELEASE}" = "1" ]; \
then pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/${TORCH_ROCM} ; \
    else pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_ROCM} ; \
fi
