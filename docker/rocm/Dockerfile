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

ARG ROCM_VERSION=6.3.4
ARG UBUNTU_VERSION=22.04

FROM rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}

# Install necessary packages
ENV PATH="/opt/rocm/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \ 
    sudo build-essential git bash-completion \
    python3.10 python3-pip python3.10-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoremove -y && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    pip install --no-cache-dir --upgrade pip setuptools wheel requests && \
    cd /opt/rocm/share/amd_smi && pip install .

RUN apt-get update && apt-get upgrade -y && apt-get install -y hipsparse hipblas hipsolver rocthrust && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoremove -y

# Install PyTorch
ARG TORCH_VERSION=""
ARG TORCH_ROCM=rocm6.3
ARG TORCH_RELEASE_TYPE=stable

RUN if [ -n "${TORCH_VERSION}" ]; then \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_ROCM} ; \
elif [ "${TORCH_RELEASE_TYPE}" = "stable" ]; then \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_ROCM} ; \
elif [ "${TORCH_RELEASE_TYPE}" = "nightly" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/${TORCH_ROCM} ; \
else \
    echo "Error: Invalid TORCH_RELEASE_TYPE. Must be 'stable', 'nightly', or specify a TORCH_VERSION." && exit 1 ; \
fi

# Install quantization libraries from source
ENV MAX_JOBS=1
ENV ROCM_VERSION=6.3
ENV GPTQMODEL_FORCE_BUILD=1
ENV PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100"

RUN pip install -v gptqmodel --no-build-isolation --no-cache-dir
