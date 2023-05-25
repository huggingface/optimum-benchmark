from typing import Any, Dict, List

from torch.fx.graph_module import GraphModule
from torch.fx import Interpreter
from torch.fx.node import Node
import torch
import time


class SymbolicProfiler(Interpreter):
    def __init__(self, graph_module: GraphModule):
        super().__init__(graph_module)

        self.nodes_latencies: Dict[Node, List[float]] = {}
        self.device: str = graph_module.device.type  # type: ignore[attr-defined]

    def run(self, *args) -> Any:
        return_val = super().run(*args)
        return return_val

    def run_node(self, node: Node) -> Any:
        if self.device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream=torch.cuda.current_stream())
            return_val = super().run_node(node)
            end.record(stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            node_runtime = start.elapsed_time(end) / 1e3
        else:
            start = time.perf_counter_ns()
            return_val = super().run_node(node)
            end = time.perf_counter_ns()
            node_runtime = (end - start) / 1e9

        self.nodes_latencies.setdefault(node, [])
        self.nodes_latencies[node].append(node_runtime)
        return return_val
