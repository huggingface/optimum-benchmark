ARG UBUNTU_VERSION=22.04

FROM ubuntu:${UBUNTU_VERSION}

# Install necessary packages
ENV PATH="/home/user/.local/bin:${PATH}"
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3-pip \
    python3.10-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    pip install --no-cache-dir --upgrade pip setuptools wheel 

# Create a non-root user
ARG GROUP_ID
ARG USER_ID

RUN addgroup --gid $GROUP_ID group
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# Change user
USER user
WORKDIR /home/user

# Install PyTorch
ARG TORCH_VERSION="stable"

RUN if [ "${TORCH_VERSION}" = "stable" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ; \
elif [ "${TORCH_VERSION}" = "nighly" ]; then \
    pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu ; \
else \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu ; \
fi
