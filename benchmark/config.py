from itertools import product
from dataclasses import dataclass, field

import torch
import logging
from typing import Optional


@dataclass
class BenchmarkConfig:
    """
    Benchmark configuration
    """

    # models space
    MODELS: list = field(default_factory=lambda: ['bert-base-uncased'])

    # inputs space
    BATCH_SIZES: list = field(default_factory=lambda: [1])
    SEQ_LENS: list = field(default_factory=lambda: [128])
    SPARSITIES: list = field(default_factory=lambda: [0])

    # env space
    DEVICES: list = field(default_factory=lambda: ['cpu'])
    THREADS: list = field(default_factory=lambda: [1])

    # a function that takes model and inputs to run the model
    run_func: callable = field(default_factory=lambda: lambda model, inputs: model(**inputs))

    # a function that takes batch size and sequence length to generate dummy inputs
    dummy_inputs_func: callable = field(default_factory=lambda: lambda bs, sl, sp: {
        'input_ids': torch.randint(0, 1000, (bs, sl)),
        'attention_mask': torch.ones((bs, sl)),
        'token_type_ids': torch.zeros((bs, sl)),
    })

    # used to control the duration of a benchmark
    total_duration: Optional[int] = None
    min_run_time: Optional[int] = None
    num_run_times: Optional[int] = None

    def __post_init__(self):

        self.models_space = self.MODELS

        self.inputs_space = list(
            product(
                self.BATCH_SIZES,
                self.SEQ_LENS,
                self.SPARSITIES,
            ))

        self.env_space = list(
            product(
                self.DEVICES,
                self.THREADS,
            ))

        # decide whether to use blocked_autorange or timeit
        if self.num_run_times is not None:
            self.use_blocked_autorange = False

        elif self.min_run_time is not None:
            self.use_blocked_autorange = True

        # min_run_time can be given in the benchmark_config or infered/overridden from total_duration and benchmark_space
        elif self.total_duration is not None:
            self.use_blocked_autorange = True
            self.min_run_time = self.total_duration // (
                len(self.inputs_space) * len(self.models_space) * len(self.env_space))

        else:
            self.use_blocked_autorange = False
            self.num_run_times = 1
            logging.warning(
                'total_duration, min_run_time or num_run_times should be given in the benchmark_config.\n'
                'num_run_times is set to 1 by default which means every configuration will be run once.')
