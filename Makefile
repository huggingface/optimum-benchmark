#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)
DEFAULT_CLONE_URL := https://github.com/huggingface/optimum-benchmark.git
# If CLONE_URL is empty, revert to DEFAULT_CLONE_URL
REAL_CLONE_URL = $(if $(CLONE_URL),$(CLONE_URL),$(DEFAULT_CLONE_URL))

# Install the library in development mode
.PHONY:	style_check style test

# Run code quality checks
style_check:
	black --check .
	ruff .

# Format the code
style:
	black .
	ruff . --fix

# Run tests for the library
test:
	python -m pytest tests