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

    hydra_dfs = {}
    config_dfs = {}
    inference_dfs = {}

    for root_folder in root_folders:
        inference_dfs.update(
            {f.parent.absolute().as_posix(): pd.read_csv(f) for f in root_folder.glob("**/training_results.csv")}
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
        hydra_dfs.update(
            {
                f.parent.parent.absolute()
                .as_posix(): pd.DataFrame.from_dict(
                    flatten(
                        OmegaConf.to_container(OmegaConf.load(f), resolve=False),
                        reducer="dot",
                    ),
                    orient="index",
                )
                .T
                for f in root_folder.glob("**/.hydra/hydra.yaml")
                if f.parent.parent.absolute().as_posix() in inference_dfs.keys()
            }
        )

    if len(inference_dfs) == 0 or len(config_dfs) == 0:
        raise ValueError(f"No results found in {root_folder}")

    # Merge inference and config dataframes
    inference_reports = [
        config_dfs[name]
        .merge(inference_dfs[name], left_index=True, right_index=True)
        .merge(hydra_dfs[name], left_index=True, right_index=True)
        for name in inference_dfs.keys()
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
        "hydra.job.env_set.CUDA_VISIBLE_DEVICES": "CUDAs",
        "benchmark.training_arguments.per_device_train_batch_size": "Per Process Batch Size",
        "benchmark.dataset_shapes.sequence_length": "Sequence Length",
        #
        "training.throughput(samples/s)": "Training Throughput (samples/s)",
    }
    short_report = full_report[list(short_columns.keys())].rename(columns=short_columns)

    short_report["GPU Name"] = short_report["GPUs"].str[0]
    short_report["Num GPUs"] = short_report["GPUs"].str.len()
    short_report["Num CUDAs"] = short_report["CUDAs"].str.split(",").str.len()
    short_report["Num Processes"] = short_report[["Num GPUs", "Num CUDAs"]].min(axis=1)

    short_report["GPU Name"].replace("NVIDIA A100-SXM4-80GB", "1xA100", inplace=True)
    short_report["GPU Name"].replace("AMD INSTINCT MI250 (MCM) OAM AC MBA", "1xMI250", inplace=True)
    short_report["GPU Name"] = short_report[["GPU Name", "Num Processes"]].apply(
        lambda x: "1xGCD-MI250" if x["GPU Name"] == "1xMI250" and x["Num Processes"] == 1 else x["GPU Name"],
        axis=1,
    )
    short_report["Effective Batch Size"] = short_report["Per Process Batch Size"] * short_report["Num Processes"]
    short_report["Group"] = short_report["GPU Name"] + "-" + short_report["Experiment Name"]
    short_report.to_csv(f"{report_folder}/short_report.csv")

    return short_report


def get_batch_plots(short_report, report_folder, plot="bar"):
    fig1, ax1 = plt.subplots()

    batch_column = "Effective Batch Size"
    short_report = short_report.sort_values(by="Group", ascending=True)
    groups = short_report["Group"].unique().tolist()
    x = np.arange(
        short_report[batch_column].min() - 1,
        len(short_report[batch_column].unique()) + (short_report[batch_column].min() - 1),
    )
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
                group_report["Training Throughput (samples/s)"],
                label=group,
                width=width,
            )
            offset += width
        elif plot == "line":
            ax1.plot(
                x_,
                group_report["Training Throughput (samples/s)"],
                label=group,
                marker="o",
            )

    ax1.set_xticks(x)
    ax1.set_ylim(bottom=0)
    ax1.set_xticklabels(short_report[batch_column].sort_values().unique().tolist())
    ax1.set_xlabel(batch_column)
    ax1.set_ylabel("Training Throughput (samples/s)")
    ax1.set_title(f"Training Throughput per Batch Size ({short_report['Model'].unique()[0]})")
    ax1.legend(fancybox=True, shadow=True)

    legend = plt.legend(loc="upper center")
    legend.get_frame().set_facecolor((0, 0, 1, 0.1))
    legend.get_frame().set_alpha(None)
    plt.tight_layout()

    fig1.savefig(f"{report_folder}/training_throughput_{plot}_plot.png")

    return fig1


def get_peak_trainong_throughput_plot(short_report, report_folder):
    # a bar plot with one bar per group, representing the max attainable throughput in tokens/s
    fig, ax = plt.subplots()

    max_training_throughput = short_report.groupby("Group")["Training Throughput (samples/s)"].max().reset_index()
    max_training_throughput = (
        short_report.merge(max_training_throughput, on=["Group", "Training Throughput (samples/s)"])
        .sort_values(by="Training Throughput (samples/s)", ascending=True)
        .reset_index()
    )

    ax.bar(
        max_training_throughput["Group"],
        max_training_throughput["Training Throughput (samples/s)"],
        color=plt.cm.Paired(np.arange(len(max_training_throughput))),
    )

    for i, v in enumerate(max_training_throughput["Effective Batch Size"]):
        ax.text(
            i,
            max_training_throughput["Training Throughput (samples/s)"].iloc[i],
            f"bs={v}",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Group")
    ax.set_ylabel("Peak Training Throughput (samples/s)")
    ax.set_title(f"Peak Training Throughput ({short_report['Model'].unique()[0]})")
    ax.set_ylim(top=max_training_throughput["Training Throughput (samples/s)"].max() * 1.1)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig(f"{report_folder}/peak_training_throughput.png")

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
        )

    _ = get_peak_trainong_throughput_plot(
        short_report,
        report_folder,
    )
    print("Report generated successfully!")


if __name__ == "__main__":
    generate_report()
