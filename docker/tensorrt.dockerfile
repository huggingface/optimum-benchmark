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

# to build with tensorrt:23.09
# docker build -f docker/tensorrt.dockerfile -t opt-bench-tensorrt:23.09 .
# to build with tensorrt:22.12
# docker build -f docker/tensorrt.dockerfile --build-arg TENSORRT_VERSION=22.12 -t opt-bench-tensorrt:22.12 .

ARG TENSORRT_VERSION=23.09

FROM nvcr.io/nvidia/tensorrt:${TENSORRT_VERSION}-py3

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Run as non-root user
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# Install and update tools to minimize security vulnerabilities
RUN apt-get update
RUN apt-get install -y software-properties-common wget apt-utils patchelf git libprotobuf-dev protobuf-compiler cmake \
    bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion libopenmpi-dev &&
    apt-get clean
RUN unattended-upgrade
RUN apt-get autoremove -y

# Add user to sudoers
RUN adduser user sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >>/etc/sudoers

# Change user
USER user
WORKDIR /home/user

# Update pip
RUN pip install --upgrade pip
