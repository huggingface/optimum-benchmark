import pandas as pd
from pathlib import Path
from pandas import DataFrame

from omegaconf import OmegaConf
from flatten_dict import flatten

import seaborn as sns
import matplotlib.pyplot as plt

from rich.table import Table
from rich.console import Console
from rich.terminal_theme import MONOKAI


def gather_inference_report(root_folder: Path) -> DataFrame:
    # key is path to inference file as string, value is dataframe
    inference_dfs = {
        f.parent.absolute().as_posix(): pd.read_csv(f)
        for f in root_folder.glob(f"**/inference_results.csv")
    }

    # key is path to config file as string, value is flattened dict
    config_dfs = {
        f.parent.absolute()
        .as_posix(): pd.DataFrame.from_dict(
            flatten(OmegaConf.load(f), reducer="dot"), orient="index"
        )
        .T
        for f in root_folder.glob(f"**/hydra_config.yaml")
        if f.parent.absolute().as_posix() in inference_dfs.keys()
    }

    if len(inference_dfs) == 0 or len(config_dfs) == 0:
        raise ValueError(f"No results found in {root_folder}")

    # Merge inference and config dataframes
    inference_reports = [
        config_dfs[name].merge(inference_dfs[name], left_index=True, right_index=True)
        for name in inference_dfs.keys()
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
    if type(element) == float:
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
    elif type(element) == bool:
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


def populate_inference_rich_table(
    table, report, with_baseline=False, with_generate=False
):
    perf_columns = [
        "forward.latency(s)",
        "forward.throughput(iter/s)",
    ]

    if with_baseline:
        perf_columns.append("forward.speedup(%)")

    if with_generate:
        perf_columns += ["generate.throughput(tok/s)"]
        if with_baseline:
            perf_columns.append("generate.speedup(%)")

    additional_columns = [
        col
        for col in report.columns
        if report[col].nunique() > 1
        and "backend" in col
        and "_target_" not in col
        and "version" not in col
    ]

    # display interesting columns in multilevel hierarchy
    display_report = report[additional_columns + perf_columns]
    display_report.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split(".")) for col in display_report.columns]
    )

    # we add a column for the index
    table.add_column("Experiment Name", justify="left", header_style="")
    # then we add the rest of the columns
    for level in range(display_report.columns.nlevels):
        columns = display_report.columns.get_level_values(level).to_list()
        for i in range(len(columns)):
            if columns[i] != columns[i]:  # nan
                columns[i] = ""

        if level < display_report.columns.nlevels - 1:
            for col in columns:
                table.add_column(col, header_style="")
            pass
        else:
            table.add_row("", *columns, end_section=True)

    # we populate the table with values
    for i, row in enumerate(display_report.itertuples(index=True)):
        if i == display_report.shape[0] - 1:
            table_row = format_row(row, style="yellow")
        else:
            table_row = format_row(row)

        table.add_row(*table_row)

    return table


def get_inference_plots(report, with_baseline=False):
    # create bar charts seperately
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    fig2, ax2 = None, None

    sns.barplot(
        x=report.index,
        y=report["forward.throughput(iter/s)"],
        ax=ax1,
        width=0.5,
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax1.set_xlabel("Experiment")
    ax1.set_ylabel("Forward Throughput (iter/s)")
    ax1.set_title("Forward Throughput by Experiment")

    if with_baseline:
        # add speedup text on top of each bar
        baselineforward_throughput = report["forward.throughput(iter/s)"].iloc[-1]
        for p in ax1.patches:
            speedup = (p.get_height() / baselineforward_throughput - 1) * 100
            ax1.annotate(
                f"{'+' if speedup>0 else '-'}{abs(speedup):.2f}%",
                (p.get_x() + p.get_width() / 2, 1.02 * p.get_height()),
                ha="center",
                va="center",
            )
        ax1.set_title("Forward Throughput and Speedup by Experiment")

    if "generate.throughput(tok/s)" in report.columns:
        fig2, ax2 = plt.subplots(figsize=(20, 10))
        sns.barplot(
            x=report.index,
            y=report["generate.throughput(tok/s)"],
            ax=ax2,
            width=0.5,
        )
        ax2.set_xticklabels(
            ax2.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        ax2.set_xlabel("Experiment")
        ax2.set_ylabel("Generate Throughput (tok/s)")
        ax2.set_title("Generate Throughput by Experiment")

        if with_baseline:
            # add speedup text on top of each bar
            baseline_generate_throughput = report["generate.throughput(tok/s)"].iloc[-1]
            for p in ax2.patches:
                speedup = (p.get_height() / baseline_generate_throughput - 1) * 100
                ax2.annotate(
                    f"{'+' if speedup>0 else '-'}{abs(speedup):.2f}%",
                    (p.get_x() + p.get_width() / 2, 1.02 * p.get_height()),
                    ha="center",
                    va="center",
                )
            ax2.set_title("Generate Throughput and Speedup by Experiment")

    return fig1, fig2


def compute_speedup(report, with_generate=False):
    # compute speedup for each experiment compared to baseline
    report["forward.speedup(%)"] = (
        report["forward.throughput(iter/s)"]
        / report["forward.throughput(iter/s)"].iloc[-1]
        - 1
    ) * 100

    if with_generate:
        report["generate.speedup(%)"] = (
            report["generate.throughput(tok/s)"]
            / report["generate.throughput(tok/s)"].iloc[-1]
            - 1
        ) * 100

    return report


def main(experiments_folders, baseline_folder=None):
    # gather experiments reports
    inference_experiments = [
        gather_inference_report(experiment) for experiment in experiments_folders
    ]
    inference_report = pd.concat(inference_experiments, axis=0)

    # sort by forward throughput
    inference_report.sort_values(
        by="forward.throughput(iter/s)", ascending=False, inplace=True
    )

    # some flags
    with_baseline = baseline_folder is not None
    with_generate = "generate.throughput(tok/s)" in inference_report.columns

    if with_baseline:
        # gather baseline report
        inference_baseline = gather_inference_report(baseline_folder)
        assert (
            inference_baseline.shape[0] == 1
        ), "baseline folder should contain only one experiment"
        # add baseline to experiment
        inference_report = pd.concat([inference_report, inference_baseline], axis=0)
        # compute speedup compared to baseline
        inference_report = compute_speedup(inference_report, with_generate)

    # there should be only one device, batch_size and new_tokens (unique triplet)
    unique_devices = inference_report["device"].unique()
    assert (
        len(unique_devices) == 1
    ), "there should be only one device (apples to apples comparison)"
    device = unique_devices[0]
    unique_batch_sizes = inference_report["benchmark.batch_size"].unique()
    assert (
        len(unique_batch_sizes) == 1
    ), "there should be only one batch_size (apples to apples comparison)"
    batch_size = unique_batch_sizes[0]

    unique_new_tokens = inference_report["benchmark.new_tokens"].unique()
    assert (
        len(unique_new_tokens) == 1
    ), "there should be only one new_tokens (apples to apples comparison)"
    new_tokens = unique_new_tokens[0]

    # create reporting directory
    reporting_directory = f"reports/{device}_{batch_size}"
    if with_generate:
        reporting_directory += f"_{new_tokens}"
    Path(reporting_directory).mkdir(exist_ok=True, parents=True)

    # Rich table
    rich_title = "Inferencing Report"
    rich_title += f"\nDevice: {device} | Batch Size: {batch_size}"
    if with_generate:
        rich_title += f" | New Tokens: {new_tokens}"

    rich_table = Table(
        show_header=True,
        title=rich_title,
        show_lines=True,
    )
    console = Console(record=True)
    rich_table = populate_inference_rich_table(
        rich_table, inference_report, with_baseline, with_generate
    )

    console.print(rich_table, justify="left", no_wrap=True)
    console.save_svg(f"{reporting_directory}/rich_table.svg", theme=MONOKAI)

    forward_fig, generate_fig = get_inference_plots(inference_report, args.baseline)
    forward_fig.savefig(f"{reporting_directory}/forward_throughput.png")
    if generate_fig is not None:
        generate_fig.savefig(f"{reporting_directory}/generate_throughput.png")

    inference_report.to_csv(f"{reporting_directory}/inference_report.csv", index=True)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--experiments",
        "-e",
        nargs="*",
        required=True,
        type=Path,
        default="experiments/",
        help="The folder containing the results of experiments.",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        required=False,
        type=Path,
        help="The folders containing the results of baseline.",
    )
    args = parser.parse_args()

    main(args.experiments, args.baseline)
