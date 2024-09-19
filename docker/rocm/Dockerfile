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
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \ 
    rocsparse-dev hipsparse-dev rocthrust-dev rocblas-dev hipblas-dev \
    sudo build-essential git bash-completion \
    python3.10 python3-pip python3.10-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    pip install --no-cache-dir --upgrade pip setuptools wheel requests && \
    cd /opt/rocm/share/amd_smi && pip install .

# Install PyTorch
ARG TORCH_ROCM=rocm5.7
ARG TORCH_RELEASE_TYPE=stable
ARG TORCH_VERSION

RUN if [ -n "${TORCH_VERSION}" ]; then \
    # Install specific torch version if TORCH_VERSION is provided
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_ROCM} ; \
elif [ "${TORCH_RELEASE_TYPE}" = "stable" ]; then \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_ROCM} ; \
elif [ "${TORCH_RELEASE_TYPE}" = "nightly" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/${TORCH_ROCM} ; \
else \
    echo "Error: Invalid TORCH_RELEASE_TYPE. Must be 'stable', 'nightly', or specify a TORCH_VERSION." && exit 1 ; \
fi
# Install quantization libraries from source
ENV ROCM_VERSION=5.7
ENV PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100"
ARG IMAGE_FLAVOR
ENV IMAGE_FLAVOR=${IMAGE_FLAVOR}

COPY scripts/install_quantization_libs.py /internal/install_quantization_libs.py
RUN 
# RUN python internal/install_quantization_libs.py --install-autogptq-from-source --install-autoawq-from-source