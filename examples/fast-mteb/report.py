from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flatten_dict import flatten
from omegaconf import OmegaConf
from pandas import DataFrame
from rich.console import Console
from rich.table import Table
from rich.terminal_theme import MONOKAI


def gather_inference_report(root_folder: Path) -> DataFrame:
    # key is path to inference file as string, value is dataframe
    inference_dfs = {
        f.parent.absolute().as_posix(): pd.read_csv(f) for f in root_folder.glob("**/inference_results.csv")
    }

    # key is path to config file as string, value is flattened dict
    config_dfs = {
        f.parent.absolute()
        .as_posix(): pd.DataFrame.from_dict(flatten(OmegaConf.load(f), reducer="dot"), orient="index")
        .T
        for f in root_folder.glob("**/hydra_config.yaml")
        if f.parent.absolute().as_posix() in inference_dfs.keys()
    }

    if len(inference_dfs) == 0 or len(config_dfs) == 0:
        raise ValueError(f"No results found in {root_folder}")

    # Merge inference and config dataframes
    inference_reports = [
        config_dfs[name].merge(inference_dfs[name], left_index=True, right_index=True) for name in inference_dfs.keys()
    ]

    # Concatenate all reports
    inference_report = pd.concat(inference_reports, axis=0, ignore_index=True)
    inference_report.set_index("experiment_name", inplace=True)
    return inference_report


def style_element(element, style=""):
    if style:
        return f"[{style}]{element}[/{style}]"
    else:
        return element


def format_element(element, style=""):
    if isinstance(element, float):
        if element != element:  # nan
            formated_element = ""
        elif abs(element) >= 1:
            formated_element = f"{element:.2f}"
        elif abs(element) > 1e-6:
            formated_element = f"{element:.2e}"
        else:
            formated_element = f"{element}"
    elif element is None:
        formated_element = ""
    elif isinstance(element, bool):
        if element:
            formated_element = style_element("✔", style="green")
        else:
            formated_element = style_element("✘", style="red")
    else:
        formated_element = str(element)

    return style_element(formated_element, style=style)


def format_row(row, style=""):
    formated_row = []
    for element in row:
        formated_row.append(format_element(element, style=style))
    return formated_row


def get_short_report(inference_report):
    short_columns = {
        "backend.name": "Backend",
        "backend.provider": "Provider",
        "benchmark.input_shapes.batch_size": "Batch Size",
        "benchmark.input_shapes.sequence_length": "Sequence Length",
        "forward.latency(s)": "Forward Latency (s)",
        "forward.throughput(samples/s)": "Forward Throughput (samples/s)",
    }
    short_report = (
        inference_report[list(short_columns.keys())]
        .rename(columns=short_columns)
        .sort_values(by=["Batch Size", "Sequence Length"], ascending=True)
    )

    short_report["Backend"] = short_report["Backend"].str.replace("pytorch", "PyTorch")
    short_report["Backend"] = short_report["Backend"].str.replace("onnxruntime", "OnnxRuntime")

    return short_report


def get_rich_table(short_report):
    # create rich table
    rich_table = Table(show_header=True, show_lines=True)
    # we add a column for the index
    rich_table.add_column("Experiment Name", justify="left", header_style="")
    # we populate the table with values
    for column in short_report.columns:
        rich_table.add_column(column, justify="right", header_style="bold")
    # we add rows
    for index, row in short_report.iterrows():
        rich_table.add_row(index, *format_row(row.values, style=""))

    return rich_table


def get_throughput_plot(short_report):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    short_report["Forward Latency (ms)"] = short_report["Forward Latency (s)"] * 1000
    short_report["Backend"] = short_report[["Backend", "Provider"]].apply(
        lambda x: f"{x.iloc[0]}+{x.iloc[1]}" if x.iloc[1] == x.iloc[1] else f"{x.iloc[0]}", axis=1
    )

    width = 0.3
    n_backends = len(short_report["Backend"].unique())
    for i, backend in enumerate(short_report["Backend"].unique(), start=-n_backends // 2):
        # for latency, we study the case of batch size 1 across all sequence lengths
        backend_report = (
            short_report[(short_report["Backend"] == backend) & (short_report["Batch Size"] == 1)]
            .drop_duplicates(subset=["Sequence Length"])
            .sort_values(by="Sequence Length", ascending=True)
        )
        seq_lens_axis = np.arange(backend_report["Sequence Length"].nunique()) + width * i
        ax1.bar(
            seq_lens_axis,
            backend_report["Forward Latency (ms)"],
            width=width,
            label=backend,
        )

        # for throughput, we study the case of sequence length 256 across all batch sizes
        backend_report = (
            short_report[
                (short_report["Backend"] == backend)
                & (short_report["Sequence Length"] == 256)
                & (short_report["Batch Size"] <= 256)
            ]
            .drop_duplicates(subset=["Batch Size"])
            .sort_values(by="Batch Size", ascending=True)
        )
        ax2.plot(
            backend_report["Batch Size"],
            backend_report["Forward Throughput (samples/s)"],
            label=backend,
            marker="o",
        )

    ax1.legend()
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Forward Latency (ms)")
    ax1.set_title("Forward Latency per Sequence Length")
    ax1.set_xticks(np.arange(len(short_report["Sequence Length"].unique())))
    ax1.set_xticklabels(short_report["Sequence Length"].unique())
    ax1.axhline(y=1, color="black", linestyle="--")
    ax1.axhline(y=2, color="red", linestyle="--")

    ax2.legend()
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Forward Throughput (samples/s)")
    ax2.set_title("Forward Throughput per Batch Size")

    return fig1, fig2


def generate_report():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiments",
        "-e",
        type=Path,
        required=True,
        help="The folder containing the results of experiments.",
    )
    parser.add_argument(
        "--report-name",
        "-r",
        type=str,
        required=False,
        help="The name of the report.",
    )

    args = parser.parse_args()
    experiments_folders = args.experiments

    if args.report_name:
        report_folder = f"artifacts/{args.report_name}"
    else:
        report_folder = "artifacts"
    Path(report_folder).mkdir(parents=True, exist_ok=True)

    # gather experiments results
    inference_report = gather_inference_report(experiments_folders)
    inference_report.sort_values(by="forward.throughput(samples/s)", ascending=False, inplace=True)
    inference_report.to_csv(f"{report_folder}/full_report.csv")

    short_report = get_short_report(inference_report)
    short_report.to_csv(f"{report_folder}/short_report.csv")

    rich_table = get_rich_table(short_report)
    console = Console(record=True)
    console.print(rich_table, justify="center")
    console.save_svg(f"{report_folder}/rich_table.svg", theme=MONOKAI, title="Inference Report")

    forward_latency_plot, forward_throughput_plot = get_throughput_plot(short_report)
    forward_latency_plot.savefig(f"{report_folder}/forward_latency_plot.png")
    forward_throughput_plot.savefig(f"{report_folder}/forward_throughput_plot.png")


if __name__ == "__main__":
    generate_report()
