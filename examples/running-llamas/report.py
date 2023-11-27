from argparse import ArgumentParser
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
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
        "benchmark.input_shapes.batch_size": "Batch Size",
        "benchmark.input_shapes.sequence_length": "Sequence Length",
        #
        "decode.latency(s)": "Decode Latency (s)",
        "forward.latency(s)": "Prefill Latency (s)",
        "generate.latency(s)": "Generate Latency (s)",
        #
        "decode.throughput(tokens/s)": "Decode Throughput (tokens/s)",
        "forward.throughput(samples/s)": "Prefill Throughput (samples/s)",
        "generate.throughput(tokens/s)": "Generate Throughput (tokens/s)",
    }
    short_report = full_report[list(short_columns.keys())].rename(columns=short_columns)

    short_report["GPU Name"] = short_report["GPUs"].str[0]
    short_report["Num GPUs"] = short_report["GPUs"].str.len().astype(str)
    short_report["GPU Name"].replace("NVIDIA A100-SXM4-80GB", "A100", inplace=True)
    short_report["GPU Name"].replace("AMD INSTINCT MI250 (MCM) OAM AC MBA", "GCD-MI250", inplace=True)

    short_report["Group"] = (
        short_report["Num GPUs"] + "x" + short_report["GPU Name"] + "-" + short_report["Experiment Name"]
    )
    short_report.to_csv(f"{report_folder}/short_report.csv")

    return short_report


import numpy as np


def get_plots(short_report, report_folder: str = "artifacts", plot="bar"):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    groups = sorted(short_report["Group"].unique().tolist())
    x = np.arange(len(short_report["Batch Size"].unique()))
    multiplier = 0
    width = 0.2

    for group in groups:
        offset = width * multiplier
        mask = short_report["Group"] == group
        group_report = short_report[mask].sort_values(by="Batch Size")

        prefill_latency = group_report[["Batch Size", "Prefill Latency (s)"]]
        decode_throughput = group_report[["Batch Size", "Decode Throughput (tokens/s)"]]
        #
        generate_latency = group_report[["Batch Size", "Generate Latency (s)"]]
        generate_throughput = group_report[["Batch Size", "Generate Throughput (tokens/s)"]]

        if plot == "bar":
            ax1.bar(
                (x + offset)[: len(prefill_latency["Batch Size"])],
                prefill_latency["Prefill Latency (s)"],
                label=group,
                width=width,
            )
            ax2.bar(
                (x + offset)[: len(decode_throughput["Batch Size"])],
                decode_throughput["Decode Throughput (tokens/s)"],
                label=group,
                width=width,
            )
            #
            ax3.bar(
                (x + offset)[: len(generate_latency["Batch Size"])],
                generate_latency["Generate Latency (s)"],
                label=group,
                width=width,
            )
            ax4.bar(
                (x + offset)[: len(generate_throughput["Batch Size"])],
                generate_throughput["Generate Throughput (tokens/s)"],
                label=group,
                width=width,
            )
        elif plot == "line":
            ax1.plot(
                prefill_latency["Batch Size"],
                prefill_latency["Prefill Latency (s)"],
                label=group,
                marker="o",
            )
            ax2.plot(
                decode_throughput["Batch Size"],
                decode_throughput["Decode Throughput (tokens/s)"],
                label=group,
                marker="o",
            )
            #
            ax3.plot(
                generate_latency["Batch Size"],
                generate_latency["Generate Latency (s)"],
                label=group,
                marker="o",
            )
            ax4.plot(
                generate_throughput["Batch Size"],
                generate_throughput["Generate Throughput (tokens/s)"],
                label=group,
                marker="o",
            )

        multiplier += 1

    if plot == "bar":
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_report["Batch Size"].sort_values().unique().tolist())

        ax2.set_xticks(x)
        ax2.set_xticklabels(short_report["Batch Size"].sort_values().unique().tolist())

        ax3.set_xticks(x)
        ax3.set_xticklabels(short_report["Batch Size"].sort_values().unique().tolist())

        ax4.set_xticks(x)
        ax4.set_xticklabels(short_report["Batch Size"].sort_values().unique().tolist())

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Prefill Latency (s)")
    ax1.set_title(f"{short_report['Model'].unique()[0]} Prefill Latency per Batch Size")

    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Decode Throughput (tokens/s)")
    ax2.set_title(f"{short_report['Model'].unique()[0]} Decode Throughput per Batch Size")

    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("Generate Latency (s)")
    ax3.set_title(f"{short_report['Model'].unique()[0]} Generate Latency per Batch Size")

    ax4.set_xlabel("Batch Size")
    ax4.set_ylabel("Generate Throughput (tokens/s)")
    ax4.set_title(f"{short_report['Model'].unique()[0]} Generate Throughput per Batch Size")

    ax1.legend(fancybox=True, shadow=True)
    ax2.legend(fancybox=True, shadow=True)
    ax3.legend(fancybox=True, shadow=True)
    ax4.legend(fancybox=True, shadow=True)

    fig1.savefig(f"{report_folder}/prefill_latency_plot.png")
    fig2.savefig(f"{report_folder}/decode_throughput_plot.png")
    # fig3.savefig(f"{report_folder}/generate_latency_plot.png")
    # fig4.savefig(f"{report_folder}/generate_throughput_plot.png")

    return fig1, fig2, fig3, fig4


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
    figs = get_plots(
        short_report,
        report_folder=report_folder,
    )


if __name__ == "__main__":
    generate_report()
