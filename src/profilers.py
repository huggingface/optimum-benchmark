from typing import Any, List, Tuple
from logging import getLogger

from optimum.onnxruntime import ORTModel
from torch.fx.graph_module import GraphModule
from torch.fx import Interpreter
from torch.fx.node import Node
import torch

import json
import time

LOGGER = getLogger("profiler")


class PytorchProfilingWrapper(Interpreter):
    def __init__(self, module: GraphModule):
        super().__init__(module)
        self.profiling_latencies: List[Tuple[str, str, float]] = []

    def run(self, *args) -> Any:
        return_val = super().run(*args)
        return return_val

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

        LOGGER.debug(f"Node {node.name} took {node_runtime} seconds")
        self.profiling_latencies.append((node.name, node.op, node_runtime))

        return return_val

    def __call__(self, **kwargs) -> Any:
        args = kwargs.values()
        return super().run(*args)

    def get_profiling_results(self) -> List[Tuple[str, str, float]]:
        return self.profiling_latencies


class ORTProfilingWrapper:
    def __init__(self, module: ORTModel):
        self.module = module
        self.profiling_latencies: List[Tuple[str, str, float]] = []

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def get_profiling_results(self, last_run: int = 0) -> List[Tuple[str, str, float]]:
        profiling_dict = self.module.model.end_profiling()  # type: ignore

        with open(profiling_dict, encoding="utf-8") as file_obj:
            profiling_data = json.load(file_obj)

        profiling_data = extract_last_run_data(profiling_data)
        profiling_latencies = extract_latencies(profiling_data)

        return profiling_latencies


def extract_latencies(data) -> List[Tuple[str, str, float]]:
    profiling_latencies = []
    for item in data:
        cat = item.get("cat")
        if cat is None:
            continue
        dur = item.get("dur")
        if dur is None:
            continue
        arg = item.get("args")
        if arg is None:
            continue
        op_name = arg.get("op_name")

        name = item["name"]

        if cat != "Kernel" and not name.endswith("kernel_time"):
            continue
        if cat in ["Kernel", "Node"]:
            profiling_latencies.append((name, op_name, dur / 1e6))

    return profiling_latencies


def extract_last_run_data(data):
    # Here we assume that the traces are properly ordered, so we can simplify the splitting logic.
    last_run_start = 0

    for i, item in enumerate(data):
        if item.get("name") == "model_run":
            last_run_start = i

    return data[last_run_start:-1]
