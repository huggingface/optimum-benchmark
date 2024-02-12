import time

from ...import_utils import is_torch_available

from transformers import LogitsProcessor

if is_torch_available():
    import torch


# TODO: uses this class for more fine-grained latency measurements in text generation
class MeasurementProcessor(LogitsProcessor):
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend

        self.latencies = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        Callback to track the time it takes to generate one batch of tokens.
        """
        self.latencies.append(time.perf_counter_ns())

        return scores
