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

# docker build -f docker/cuda.dockerfile -t opt-bench-cuda:12.1.1-cudnn8 .
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Install and update tools to minimize security vulnerabilities
RUN apt-get update
RUN apt-get install -y software-properties-common wget apt-utils patchelf git libprotobuf-dev protobuf-compiler cmake \
    bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion libopenmpi-dev && \
    apt-get clean
RUN unattended-upgrade
RUN apt-get autoremove -y

# Install python
RUN apt-get install -y python3 python3-pip python3-dev python3-setuptools python3-wheel python3-venv && \
    apt-get clean