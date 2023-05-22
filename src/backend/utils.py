from typing import Any, Dict, List
import pandas as pd
import json
import time

from torch.fx.graph_module import GraphModule
from torch.fx import Interpreter
from torch.fx.node import Node
import torch


class SymbolicProfiler(Interpreter):
    def __init__(self, graph_module: GraphModule):
        super().__init__(graph_module)

        self.model_latencies: List[float] = []
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


def shape_to_string(shape):
    res = ""
    for dict_obj in shape:
        if len(dict_obj) > 1:
            raise ValueError("Unhandled type in _shape_to_string()")
        key = list(dict_obj.keys())[0]
        value = list(dict_obj.values())[0]
        if len(res) != 0:
            res += ","
        res += f'{key}({"x".join(str(v) for v in value)})'
    return res


def json_to_df(data):
    entries = []

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
        if name.endswith("kernel_time"):
            most_recent_kernel_launch_event = item

        block_x = arg.get("block_x", -1)
        block_y = arg.get("block_y", -1)
        block_z = arg.get("block_z", -1)
        grid_x = arg.get("grid_x", -1)
        grid_y = arg.get("grid_y", -1)
        grid_z = arg.get("grid_z", -1)

        if cat in ["Kernel", "Node"]:
            entries.append(
                {
                    "Kernel name": name,
                    "Op name": op_name,
                    "Kernel latency": dur,
                }
            )

    return pd.DataFrame(entries)


def split_data_across_runs(data, start=1, end=None):
    """
    Splits the traces according to model runs they belong to.
    By default, we skip the first model run (run 0) and consider all subsequent runs.
    """
    # Here we assume that the traces are properly ordered, so we can simplify the splitting logic.
    model_run_splits = [
        i for i, item in enumerate(data) if item.get("name") == "model_run"
    ]
    if not model_run_splits:
        print(
            'WARNING: Could not find "model_run" event in trace. Using entire traces.'
        )
        return data
    total_num_runs = len(model_run_splits)
    print(f"Found {total_num_runs} model_run events in trace.")

    assert -total_num_runs <= start < total_num_runs, f"Invalid start index {start}."
    if start < 0:
        start += total_num_runs
    if end is None:
        end = total_num_runs
    else:
        assert -total_num_runs <= end < total_num_runs, f"Invalid end index {end}."
        if end < 0:
            end += total_num_runs
    num_runs = end - start
    assert num_runs > 0, "No valid model runs are included in the split."
    print(f"Analyzing {num_runs} model run(s): {start}-{end - 1}.")

    # Add index 0 in case user wants to include the first model run.
    model_run_splits = [0, *model_run_splits]
    return data[model_run_splits[start] : model_run_splits[end]], num_runs


def load_json(profile_path):
    with open(profile_path, encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if isinstance(data, dict):
        data = data["traceEvents"]
    return data
