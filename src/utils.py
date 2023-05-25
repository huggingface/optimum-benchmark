import pandas as pd
import json


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
