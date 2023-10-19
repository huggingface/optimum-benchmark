ARG ROCM_VERSION=5.6.1
ARG UBUNTU_VERSION=22.04

FROM rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}

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
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >>/etc/sudoers

# Fix permissions
RUN usermod -g video user
RUN usermod -a -G render user

# Change user
USER user
WORKDIR /home/user

# Install PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
