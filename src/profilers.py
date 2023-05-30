from typing import Any, List, Tuple
from logging import getLogger

from optimum.onnxruntime import ORTModel
import pandas as pd
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
        self.profiling_records: List[Tuple[str, str, float]] = []

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
        self.profiling_records.append((node.name, node.op, node_runtime))

        return return_val

    def __call__(self, **kwargs) -> Any:
        args = kwargs.values()
        return super().run(*args)

    def get_profiling_records(self) -> List[Tuple[str, str, float]]:
        return self.profiling_records


class ORTProfilingWrapper:
    def __init__(self, module: ORTModel):
        self.module = module
        self.profiling_records: List[Tuple[str, str, float]] = []

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def get_profiling_records(self) -> List[Tuple[str, str, float]]:
        profiling_json = self.module.model.end_profiling()  # type: ignore
        print(profiling_json)
        with open(profiling_json) as file_obj:
            profiling_data = json.load(file_obj)
            print("data")
            if isinstance(profiling_data, dict):
                profiling_data = profiling_data["traceEvents"]

        print("extracting records")
        profiling_records = extract_last_run_records(profiling_data)
        profiling_records = normalize_records(profiling_records)

        return profiling_records


def normalize_records(data) -> List[Tuple[str, str, float]]:
    records = []
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
            records.append((name.replace("_kernel_time", ""), op_name, dur / 1e6))

    return records


def extract_last_run_records(data):
    # Here we assume that the traces are properly ordered, so we can simplify the splitting logic.
    return (
        pd.DataFrame(data)[["name", "cat", "dur", "args"]]
        .groupby("name")
        .last()
        .reset_index()
        .to_dict(orient="records")
    )
