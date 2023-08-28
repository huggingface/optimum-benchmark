import json
from logging import getLogger
from typing import List, Tuple

import pandas as pd
from optimum.onnxruntime import ORTModel

LOGGER = getLogger("ort_profiler")


class ORTProfilingWrapper:
    def __init__(self, module: ORTModel):
        self.module = module
        self.profiling_records: List[Tuple[str, str, float]] = []

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def get_profiling_records(self) -> List[Tuple[str, str, float]]:
        profiling_json = self.module.model.end_profiling()  # type: ignore
        with open(profiling_json) as file_obj:
            profiling_data = json.load(file_obj)
            if isinstance(profiling_data, dict):
                profiling_data = profiling_data["traceEvents"]

        profiling_records = extract_last_run_records(profiling_data)
        return normalize_records(profiling_records)


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
            LOGGER.debug(f"Kernel/Node {name} took {dur / 1e6:.2e} seconds")
            records.append((name.replace("_kernel_time", ""), op_name, dur / 1e6))

    return records


def extract_last_run_records(data):
    # Here we assume that the traces are properly ordered, so we can simplify the splitting logic.
    return (
        pd.DataFrame(data)[["name", "cat", "dur", "args"]]
        .groupby("name")
        .last()  # not sure if this is the right way to do it
        .reset_index()
        .to_dict(orient="records")
    )
