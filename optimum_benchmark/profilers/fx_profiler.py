import time
from logging import getLogger
from typing import Any, List, Tuple

import torch
from torch.fx import Interpreter
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

LOGGER = getLogger("fx_profiler")


class FXProfilingWrapper(Interpreter):
    def __init__(self, module: GraphModule):
        super().__init__(module)
        self.profiling_records: List[Tuple[str, str, float]] = []

    def run(self, *args) -> Any:
        return super().run(*args)

    def run_node(self, node: Node) -> Any:
        if self.module.device.type == "cuda":
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

        LOGGER.debug(f"Node {node.name} took {node_runtime:.2e} seconds")
        self.profiling_records.append((node.name, node.op, node_runtime))

        return return_val

    def __call__(self, **kwargs) -> Any:
        args = kwargs.values()
        return super().run(*args)

    def get_profiling_records(self) -> List[Tuple[str, str, float]]:
        return self.profiling_records
