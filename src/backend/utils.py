from typing import Any, Dict, List
import time
import torch

from torch.fx import Interpreter, GraphModule
from torch.fx.node import Node


class SymbolicProfiler(Interpreter):
    """
    A subclass of ``Interpreter`` that records the runtime of each node
    in the model. This is useful for profiling the runtime of a model
    and identifying bottlenecks.

    modified from the original on torch.fx.experimental.fx_profiling

    Args:
        graph_module (GraphModule): the GraphModule to interpret
    """

    def __init__(self, graph_module: GraphModule, device: str = 'cpu'):
        super().__init__(graph_module)
        self.device: str = device
        self.model_latencies: List[float] = []
        self.nodes_latencies: Dict[Node, List[float]] = {}

    def run(self, *args) -> Any:
        if self.device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return_val = super().run(*args)
            end.record()
            torch.cuda.synchronize()

            module_runtime = start.elapsed_time(end) / 1e3

        else:
            start = time.perf_counter_ns()
            return_val = super().run(*args)
            end = time.perf_counter_ns()

            module_runtime = (end - start) / 1e9

        self.model_latencies.append(module_runtime)
        return return_val

    def run_node(self, n: Node) -> Any:
        if self.device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return_val = super().run_node(n)
            end.record()
            torch.cuda.synchronize()

            node_runtime = start.elapsed_time(end) / 1e3

        else:
            start = time.perf_counter_ns()
            return_val = super().run_node(n)
            end = time.perf_counter_ns()

            node_runtime = (end - start) / 1e9

        self.nodes_latencies.setdefault(n, [])
        self.nodes_latencies[n].append(node_runtime)
        return return_val
