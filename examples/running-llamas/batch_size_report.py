from argparse import ArgumentParser
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flatten_dict import flatten
from omegaconf import OmegaConf
from pandas import DataFrame


def gather_full_report(root_folders: List[Path], report_folder: str = "artifacts") -> DataFrame:
    # key is path to inference file as string, value is dataframe

    config_dfs = {}
    inference_dfs = {}

    for root_folder in root_folders:
        inference_dfs.update(
            {f.parent.absolute().as_posix(): pd.read_csv(f) for f in root_folder.glob("**/inference_results.csv")}
        )
        config_dfs.update(
            {
                f.parent.absolute()
                .as_posix(): pd.DataFrame.from_dict(flatten(OmegaConf.load(f), reducer="dot"), orient="index")
                .T
                for f in root_folder.glob("**/hydra_config.yaml")
                if f.parent.absolute().as_posix() in inference_dfs.keys()
            }
        )

    if len(inference_dfs) == 0 or len(config_dfs) == 0:
        raise ValueError(f"No results found in {root_folder}")

    # Merge inference and config dataframes
    inference_reports = [
        config_dfs[name].merge(inference_dfs[name], left_index=True, right_index=True) for name in inference_dfs.keys()
    ]

    # Concatenate all reports
    inference_report = pd.concat(inference_reports, axis=0, ignore_index=True)
    inference_report.to_csv(f"{report_folder}/full_report.csv")

    return inference_report


def get_short_report(full_report, report_folder: str = "artifacts"):
    short_columns = {
        "model": "Model",
        "environment.gpus": "GPUs",
        "experiment_name": "Experiment Name",
        "benchmark.input_shapes.batch_size": "Per Process Batch Size",
        "benchmark.input_shapes.sequence_length": "Sequence Length",
        #
        "decode.latency(s)": "Decode Latency (s)",
        "forward.latency(s)": "Prefill Latency (s)",
        #
        "decode.throughput(tokens/s)": "Decode Throughput (tokens/s)",
        "forward.throughput(samples/s)": "Prefill Throughput (samples/s)",
        #
        "forward.max_memory_allocated(MB)": "Prefill Memory (MB)",
        "generate.max_memory_allocated(MB)": "Decode Memory (MB)",
    }
    short_report = full_report[list(short_columns.keys())].rename(columns=short_columns)

    short_report["GPU Name"] = short_report["GPUs"].str[0]
    short_report["Num GPUs"] = short_report["GPUs"].str.len()
    short_report["GPU Name"].replace("NVIDIA A100-SXM4-80GB", "A100", inplace=True)
    short_report["GPU Name"].replace("AMD INSTINCT MI250 (MCM) OAM AC MBA", "GCD-MI250", inplace=True)
    short_report["Effective Batch Size"] = short_report["Per Process Batch Size"] * short_report["Num GPUs"]

    short_report["Group"] = (
        short_report["Num GPUs"].astype(str) + "x" + short_report["GPU Name"] + "-" + short_report["Experiment Name"]
    )

    short_report.to_csv(f"{report_folder}/short_report.csv")

    return short_report


def get_plots(short_report, report_folder, batch="effective", plot="bar"):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    batch_column = "Effective Batch Size" if batch == "effective" else "Per Process Batch Size"

    groups = sorted(short_report["Group"].unique().tolist())
    x = np.arange(len(short_report[batch_column].unique()))
    width = 0.8 / len(groups)
    offset = -(width * (len(groups) - 1) / 2)

    for group in groups:
        mask = short_report["Group"] == group
        group_report = short_report[mask].sort_values(by=batch_column)

        prefill_latency = group_report[[batch_column, "Prefill Latency (s)"]]
        decode_throughput = group_report[[batch_column, "Decode Throughput (tokens/s)"]]

        if plot == "bar":
            x_ = np.arange(
                prefill_latency[batch_column].min() - 1,
                len(prefill_latency[batch_column].unique()) + (prefill_latency[batch_column].min() - 1),
            )
            ax1.bar(
                x_ + offset,
                prefill_latency["Prefill Latency (s)"],
                label=group,
                width=width,
            )
            x_ = np.arange(
                decode_throughput[batch_column].min() - 1,
                len(decode_throughput[batch_column].unique()) + (decode_throughput[batch_column].min() - 1),
            )
            ax2.bar(
                x_ + offset,
                decode_throughput["Decode Throughput (tokens/s)"],
                label=group,
                width=width,
            )
            offset += width
        elif plot == "line":
            x_ = np.arange(
                prefill_latency[batch_column].min() - 1,
                len(prefill_latency[batch_column].unique()) + (prefill_latency[batch_column].min() - 1),
            )
            ax1.plot(
                x_,
                prefill_latency["Prefill Latency (s)"],
                label=group,
                marker="o",
            )
            x_ = np.arange(
                decode_throughput[batch_column].min() - 1,
                len(decode_throughput[batch_column].unique()) + (decode_throughput[batch_column].min() - 1),
            )
            ax2.plot(
                x_,
                decode_throughput["Decode Throughput (tokens/s)"],
                label=group,
                marker="o",
            )

    ax1.set_xticks(x)
    ax1.set_ylim(bottom=0)
    ax1.set_xticklabels(short_report[batch_column].sort_values().unique().tolist())

    ax2.set_xticks(x)
    ax2.set_ylim(bottom=0)
    ax2.set_xticklabels(short_report[batch_column].sort_values().unique().tolist())

    ax1.set_xlabel(batch_column)
    ax1.set_ylabel("Prefill Latency (s)")
    ax1.set_title(f"Prefill Latency per Batch Size ({short_report['Model'].unique()[0]})")

    ax2.set_xlabel(batch_column)
    ax2.set_ylabel("Effective Decode Throughput (tokens/s)")
    ax2.set_title(f"Decode Throughput per Batch Size ({short_report['Model'].unique()[0]})")

    ax1.legend(fancybox=True, shadow=True)
    ax2.legend(fancybox=True, shadow=True)

    fig1.savefig(f"{report_folder}/prefill_latency_{plot}_plot_{batch}.png")
    fig2.savefig(f"{report_folder}/decode_throughput_{plot}_plot_{batch}.png")

    return fig1, fig2


def generate_report():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiments-folders",
        "-e",
        type=Path,
        nargs="+",
        required=True,
        help="The folder containing the results of experiments.",
    )
    parser.add_argument(
        "--report-name",
        "-r",
        type=str,
        required=False,
        default="artifacts",
        help="The name of the report.",
    )

    args = parser.parse_args()
    report_folder = args.report_name
    experiments_folders = args.experiments_folders

    Path(report_folder).mkdir(parents=True, exist_ok=True)

    # gather experiments results
    full_report = gather_full_report(
        root_folders=experiments_folders,
        report_folder=report_folder,
    )
    short_report = get_short_report(
        full_report,
        report_folder=report_folder,
    )
    for plot in ["bar", "line"]:
        for batch in ["effective", "per_process"]:
            _ = get_plots(
                short_report,
                report_folder,
                batch=batch,
                plot=plot,
            )


if __name__ == "__main__":
    generate_report()
