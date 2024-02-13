import time
from typing import List

import torch
from transformers import TrainerCallback


class LatencyTrainerCallback(TrainerCallback):
    def __init__(self, device: str, backend: str) -> None:
        self.device = device
        self.backend = backend
        self.all_latencies_list = []

    def on_step_begin(self, *args, **kwargs):
        # one record per step
        if self.device == "cuda" and self.backend == "pytorch":
            self.all_latencies_list.append(torch.cuda.Event(enable_timing=True))
            self.all_latencies_list[-1].record()
        else:
            self.all_latencies_list.append(time.perf_counter_ns())

    def on_train_end(self, *args, **kwargs):
        # one last record to measure the time of the last step
        if self.device == "cuda" and self.backend == "pytorch":
            self.all_latencies_list.append(torch.cuda.Event(enable_timing=True))
            self.all_latencies_list[-1].record()
        else:
            self.all_latencies_list.append(time.perf_counter_ns())

    def get_latencies_list(self) -> List[float]:
        if self.device == "cuda" and self.backend == "pytorch":
            torch.cuda.synchronize()  # synchronize the device to make sure all events have been recorded
            latencies_list = [
                self.all_latencies_list[i - 1].elapsed_time(self.all_latencies_list[i]) * 1e-3
                for i in range(1, len(self.all_latencies_list))
            ]
        else:
            latencies_list = [
                (self.all_latencies_list[i] - self.all_latencies_list[i - 1]) * 1e-9
                for i in range(1, len(self.all_latencies_list))
            ]

        return latencies_list
