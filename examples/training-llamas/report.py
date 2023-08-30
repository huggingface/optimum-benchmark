from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from flatten_dict import flatten
from omegaconf import OmegaConf
from pandas import DataFrame
from rich.console import Console
from rich.table import Table
from rich.terminal_theme import MONOKAI


def gather_training_report(root_folder: Path) -> DataFrame:
    # key is path to training file as string, value is dataframe
    training_dfs = {
        f.parent.absolute().as_posix(): pd.read_csv(f) for f in root_folder.glob("**/training_results.csv")
    }

    # key is path to config file as string, value is flattened dict
    config_dfs = {
        f.parent.absolute()
        .as_posix(): pd.DataFrame.from_dict(flatten(OmegaConf.load(f), reducer="dot"), orient="index")
        .T
        for f in root_folder.glob("**/hydra_config.yaml")
        if f.parent.absolute().as_posix() in training_dfs.keys()
    }

    if len(training_dfs) == 0 or len(config_dfs) == 0:
        raise ValueError(f"No results found in {root_folder}")

    # Merge training and config dataframes
    training_reports = [
        config_dfs[name].merge(training_dfs[name], left_index=True, right_index=True) for name in training_dfs.keys()
    ]

    # Concatenate all reports
    training_report = pd.concat(training_reports, axis=0, ignore_index=True)
    training_report.set_index("experiment_name", inplace=True)
    return training_report


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

def get_short_report(training_report):
    short_columns = {
        "backend.quantization_strategy": "Quantization Scheme",
        "benchmark.training_arguments.per_device_train_batch_size": "Batch Size",
        "training.throughput(samples/s)": "Training Throughput (samples/s)",
        "environment.gpu": "GPU",
    }
    short_report = training_report[list(short_columns.keys())].rename(columns=short_columns)

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
    # for each quantization scheme we plot the throughput vs batch size
    fig, ax = plt.subplots()

    for quantization_scheme in short_report["Quantization Scheme"].unique():
        mask = short_report["Quantization Scheme"] == quantization_scheme
        ax.plot(
            short_report[mask]["Batch Size"],
            short_report[mask]["Training Throughput (samples/s)"],
            label=quantization_scheme,
            marker="o",
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Training Throughput (samples/s)")
    ax.set_title("Training Throughput per Batch Size")
    ax.legend()

    return fig


def generate_report():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiments",
        "-e",
        nargs="*",
        type=Path,
        required=True,
        help="The folder containing the results of experiments.",
    )
    parser.add_argument(
        "--report-name",
        "-n",
        type=str,
        required=False,
        help="The name of the report.",
    )

    args = parser.parse_args()
    experiments_folders = args.experiments

    # gather experiments results
    training_experiments = [gather_training_report(experiment) for experiment in experiments_folders]
    training_report = pd.concat(training_experiments, axis=0)
    training_report.sort_values(by="training.throughput(samples/s)", ascending=False, inplace=True)
    training_report.to_csv("artifacts/full_report.csv")

    short_report = get_short_report(training_report)
    short_report.to_csv("artifacts/short_report.csv")

    throughput_plot = get_throughput_plot(short_report)
    throughput_plot.savefig("artifacts/throughput_plot.png")

    rich_table = get_rich_table(short_report)
    console = Console(record=True)
    console.print(rich_table, justify="center")
    console.save_svg("artifacts/rich_table.svg", theme=MONOKAI, title="Training Report")


if __name__ == "__main__":
    generate_report()
