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
        "generate.max_memory_allocated(MB)": "Generate Max Memory Allocated (MB)",
        "generate.max_memory_reserved(MB)": "Generate Max Memory Reserved (MB)",
    }
    short_report = full_report[list(short_columns.keys())].rename(columns=short_columns)

    short_report["GPU Name"] = short_report["GPUs"].str[0]
    short_report["Num GPUs"] = short_report["GPUs"].str.len()
    short_report["GPU Name"].replace("NVIDIA A100-SXM4-80GB", "1xA100", inplace=True)
    short_report["GPU Name"].replace("AMD INSTINCT MI250 (MCM) OAM AC MBA", "1xMI250", inplace=True)
    short_report["Effective Batch Size"] = short_report["Per Process Batch Size"] * short_report["Num GPUs"]
    short_report["Group"] = short_report["GPU Name"] + "-" + short_report["Experiment Name"]
    short_report.to_csv(f"{report_folder}/short_report.csv")

    return short_report


def get_batch_plots(short_report, report_folder, plot="bar", memory=True):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    batch_column = "Effective Batch Size"
    short_report = short_report.sort_values(by="Group", ascending=True)
    groups = short_report["Group"].unique().tolist()
    x = np.arange(len(short_report[batch_column].unique()))
    width = 0.8 / len(short_report["Group"].unique().tolist())
    offset = -(width * (len(groups) - 1) / 2)

    for group in groups:
        mask = short_report["Group"] == group
        group_report = short_report[mask].sort_values(by=batch_column)
        x_ = np.arange(
            group_report[batch_column].min() - 1,
            len(group_report[batch_column].unique()) + (group_report[batch_column].min() - 1),
        )
        if plot == "bar":
            ax1.bar(
                x_ + offset,
                group_report["Prefill Latency (s)"],
                label=group,
                width=width,
            )
            ax2.bar(
                x_ + offset,
                group_report["Decode Throughput (tokens/s)"],
                label=group,
                width=width,
            )
            ax3.bar(
                x_ + offset,
                group_report["Generate Max Memory Allocated (MB)"],
                label=group,
                width=width,
            )
            ax4.bar(
                x_ + offset,
                group_report["Generate Max Memory Reserved (MB)"],
                label=group,
                width=width,
            )
            offset += width
        elif plot == "line":
            ax1.plot(
                x_,
                group_report["Prefill Latency (s)"],
                label=group,
                marker="o",
            )
            ax2.plot(
                x_,
                group_report["Decode Throughput (tokens/s)"],
                label=group,
                marker="o",
            )
            ax3.plot(
                x_,
                group_report["Generate Max Memory Allocated (MB)"],
                label=group,
                marker="o",
            )
            ax4.plot(
                x_,
                group_report["Generate Max Memory Reserved (MB)"],
                label=group,
                marker="o",
            )

    ax1.set_xticks(x)
    ax1.set_ylim(bottom=0)
    ax1.set_xticklabels(short_report[batch_column].sort_values().unique().tolist())
    ax1.set_xlabel(batch_column)
    ax1.set_ylabel("Prefill Latency (s)")
    ax1.set_title(f"Prefill Latency per Batch Size ({short_report['Model'].unique()[0]})")
    ax1.legend(fancybox=True, shadow=True)

    ax2.set_xticks(x)
    ax2.set_ylim(bottom=0)
    ax2.set_xticklabels(short_report[batch_column].sort_values().unique().tolist())
    ax2.set_xlabel(batch_column)
    ax2.set_ylabel("Effective Decode Throughput (tokens/s)")
    ax2.set_title(f"Decode Throughput per Batch Size ({short_report['Model'].unique()[0]})")
    ax2.legend(fancybox=True, shadow=True)

    ax3.set_xticks(x)
    ax3.set_ylim(bottom=0)
    ax3.set_xticklabels(short_report[batch_column].sort_values().unique().tolist())
    ax3.set_xlabel(batch_column)
    ax3.set_ylabel("Generate Max Memory Allocated (MB)")
    ax3.set_title(f"Generate Max Memory Allocated per Batch Size ({short_report['Model'].unique()[0]})")
    ax3.legend(fancybox=True, shadow=True)

    ax4.set_xticks(x)
    ax4.set_ylim(bottom=0)
    ax4.set_xticklabels(short_report[batch_column].sort_values().unique().tolist())
    ax4.set_xlabel(batch_column)
    ax4.set_ylabel("Generate Max Memory Reserved (MB)")
    ax4.set_title(f"Generate Max Memory Reserved per Batch Size ({short_report['Model'].unique()[0]})")
    ax4.legend(fancybox=True, shadow=True)

    legend = plt.legend(loc="upper center")
    legend.get_frame().set_facecolor((0, 0, 1, 0.1))
    legend.get_frame().set_alpha(None)
    plt.tight_layout()

    fig1.savefig(f"{report_folder}/prefill_latency_{plot}_plot.png")
    fig2.savefig(f"{report_folder}/decode_throughput_{plot}_plot.png")

    if memory:
        fig3.savefig(f"{report_folder}/generate_max_memory_allocated_{plot}_plot.png")
        fig4.savefig(f"{report_folder}/generate_max_memory_reserved_{plot}_plot.png")
        return fig1, fig2, fig3, fig4

    return fig1, fig2


def get_peak_decode_throughput_plot(short_report, report_folder):
    # a bar plot with one bar per group, representing the max attainable throughput in tokens/s
    fig, ax = plt.subplots()

    #
    max_decode_throughput = short_report.groupby("Group")["Decode Throughput (tokens/s)"].max().reset_index()
    max_decode_throughput = (
        short_report.merge(max_decode_throughput, on=["Group", "Decode Throughput (tokens/s)"])
        .sort_values(by="Decode Throughput (tokens/s)", ascending=True)
        .reset_index()
    )

    ax.bar(
        max_decode_throughput["Group"],
        max_decode_throughput["Decode Throughput (tokens/s)"],
        color=plt.cm.Paired(np.arange(len(max_decode_throughput))),
    )

    # add batch size on top of each bar
    for i, v in enumerate(max_decode_throughput["Effective Batch Size"]):
        ax.text(
            i,
            max_decode_throughput["Decode Throughput (tokens/s)"].iloc[i],
            f"bs={v}",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Group")
    ax.set_ylabel("Peak Decode Throughput (tokens/s)")
    ax.set_title(f"Peak Decode Throughput ({short_report['Model'].unique()[0]})")
    ax.set_ylim(top=max_decode_throughput["Decode Throughput (tokens/s)"].max() * 1.1)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig(f"{report_folder}/peak_decode_throughput_bar_plot.png")

    return fig


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
        _ = get_batch_plots(
            short_report,
            report_folder,
            plot=plot,
            memory=True,
        )

    _ = get_peak_decode_throughput_plot(
        short_report,
        report_folder,
    )
    print("Report generated successfully!")


if __name__ == "__main__":
    generate_report()
