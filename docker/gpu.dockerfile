#!/usr/bin/env python
# coding=utf-8
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

# use version with CUDA 11.8 and TensorRT 8.5.1.7 to match ORT 1.14 requirements
FROM nvcr.io/nvidia/tensorrt:22.12-py3

# Ignore interactive questions during `docker build`
ENV DEBIAN_FRONTEND noninteractive

# Install and update tools to minimize security vulnerabilities
RUN apt-get update
RUN apt-get install -y software-properties-common wget apt-utils patchelf git libprotobuf-dev protobuf-compiler cmake \
    bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion libopenmpi-dev && \
    apt-get clean
RUN unattended-upgrade
RUN apt-get autoremove -y
<<<<<<< HEAD
RUN pip install --upgrade pip

# this line forces the docker build to rebuild from this point on
ARG CACHEBUST=1

# Install optimum-benchmark dependencies
COPY gpu_requirements.txt /tmp/gpu_requirements.txt
RUN pip install -r /tmp/gpu_requirements.txt
=======
RUN pip install --upgrade pip
>>>>>>> seperated backend-device tests and update setup.py
