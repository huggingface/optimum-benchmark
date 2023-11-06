from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def plot(
    report: pd.DataFrame, x_axis: str = "Batch Size", y_axis: str = "Forward Latency (s)", groupby: str = "Experiment"
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()

    for group, sweep in report.groupby(groupby):
        sorted_sweep = sweep.sort_values(by=x_axis)
        ax.plot(sorted_sweep[x_axis], sorted_sweep[y_axis], label=group, marker="o")

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{y_axis} per {x_axis}")
    ax.legend(fancybox=True, shadow=True)

    return fig, ax


def plot_cli() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        default=Path("artifacts/full_report.csv"),
        help="Path to the report csv file",
    )
    parser.add_argument(
        "--x-axis",
        "-x",
        default="Batch Size",
        help="X axis column",
    )
    parser.add_argument(
        "--y-axis",
        "-y",
        default="Forward Latency (s)",
        help="Y axis column",
    )
    parser.add_argument(
        "--groupby",
        "-g",
        default="Experiment",
        help="Groupby column",
    )
    parser.add_argument(
        "--save-file",
        "-s",
        type=Path,
        default=Path("artifacts/plot.png"),
        help="Path to the plot file",
    )

    args = parser.parse_args()
    report = pd.read_csv(args.report)
    fig, _ = plot(report, args.x_axis, args.y_axis, args.groupby)
    if args.save_file is not None:
        args.save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_file)
