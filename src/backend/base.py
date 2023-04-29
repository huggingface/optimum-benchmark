#  Copyright 2021 Hugging Face Inc.
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

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Dict, Generic, TypeVar, ClassVar, Tuple

import torch

from benchmark.base import Benchmark
from backend.config import BackendConfig

import numpy as np

LOGGER = getLogger('backend')
BackendConfigT = TypeVar('BackendConfigT', bound=BackendConfig)
TensorT = TypeVar('TensorT', torch.Tensor, np.ndarray)


class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    def __init__(self, model: str, task: str):
        self.model = model
        self.task = task

    @abstractmethod
    def execute(self, config: 'BenchmarkConfig') -> Tuple[Benchmark, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def get_dummy_inputs(self, config: 'BenchmarkConfig') -> Dict[str, TensorT]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def allocate(cls, config: 'BenchmarkConfig') -> 'Backend':
        raise NotImplementedError()

    @abstractmethod
    def configure(self, config: BackendConfigT) -> None:
        raise NotImplementedError()