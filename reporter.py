import pandas as pd
import seaborn as sns
from pathlib import Path
from pandas import DataFrame
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from flatten_dict import flatten
from argparse import ArgumentParser

from rich.console import Console
from rich.table import Table


def gather_inference_report(folder: Path) -> DataFrame:
    stats_files = [
        stats_file for stats_file in folder.glob(f"**/inference_results.csv")
    ]
    stats_dfs = {i: pd.read_csv(f, index_col=0) for i, f in enumerate(stats_files)}

    configs_files = [
        config_file for config_file in folder.glob(f"**/hydra_config.yaml")
    ]
    # only leave config files that have a corresponding stats file
    configs_files = [
        config_file
        for config_file in configs_files
        if config_file.parent in [stats_file.parent for stats_file in stats_files]
    ]
    config_dicts = {
        i: flatten(OmegaConf.load(f), reducer="dot")
        for i, f in enumerate(configs_files)
    }
    # for now there's a problem with list of operators to quantize
    for d in config_dicts.values():
        d.pop("backend.quantization_config.operators_to_quantize", None)

    configs_dfs = {i: pd.DataFrame(d, index=[0]) for i, d in config_dicts.items()}

    if len(stats_dfs) == 0 or len(configs_dfs) == 0:
        raise ValueError(f"No results found in {folder}")

    # Merge perfs dataframes with configs
    inference_reports = {
        name: configs_dfs[name].merge(
            stats_dfs[name], left_index=True, right_index=True
        )
        for name in stats_dfs.keys()
    }
    # Concatenate all reports
    inference_report = pd.concat(inference_reports.values(), axis=0, ignore_index=True)
    # inference_report["Config Path"] = configs_files  # for console display (clickable)
    inference_report.set_index("experiment_name", inplace=True)
    # sort by throughput and remove failed experiments
    inference_report.sort_values(
        by=["Model.Throughput(iter/s)"], ascending=False, inplace=True
    )

    return inference_report


def show_inference_report(report, with_baseline=False):
    # columns to display
    show_report = report[
        [
            "backend.auto_quantization",
            "backend.auto_optimization",
        ]
        + ["Model.Latency(s)", "Model.Throughput(iter/s)"]
        + (
            ["Model.Peak_Memory(MB)"]
            if "Model.Peak_Memory(MB)" in report.columns
            else []
        )
        + (["Model.Speedup(%)"] if with_baseline else [])
        + (
            ["Generation.Throughput(tok/s)"]
            if "Generation.Throughput(tok/s)" in report.columns
            else []
        )
        + (
            ["Generation.Speedup(%)"]
            if "Generation.Throughput(tok/s)" in report.columns and with_baseline
            else []
        )
    ]

    if with_baseline:
        show_report.sort_values(by="Model.Speedup(%)", inplace=True, ascending=False)

    table = Table(
        show_header=True,
        title="Inference Benchmark Report",
    )

    show_report.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split(".")) for col in show_report.columns.to_list()]
    )

    table.add_column("Experiment Name", justify="left")
    for level in range(show_report.columns.nlevels):
        columns = show_report.columns.get_level_values(level).to_list()
        for i in range(len(columns)):
            if columns[i] != columns[i]:  # nan
                columns[i] = ""

        if level < show_report.columns.nlevels - 1:
            for col in columns:
                table.add_column(col)
            pass
        else:
            table.add_row("", *columns, end_section=True)

    for row in show_report.itertuples(index=True):
        table_row = []
        for elm in row:
            if type(elm) == float:
                if abs(elm) >= 1:
                    table_row.append(f"{elm:.2f}")
                elif abs(elm) > 1e-3:
                    table_row.append(f"{elm:.2e}")
                elif elm != elm:
                    table_row.append("")
                else:
                    table_row.append(f"{elm}")
            elif elm is None:
                table_row.append("")

            elif type(elm) == bool:
                if elm:
                    table_row.append("[green]✔[/green]")
                else:
                    table_row.append("[red]✘[/red]")

            elif type(elm) == str and elm.endswith("baseline"):
                table_row.append(f"[bold][yellow]{elm}[/yellow][/bold]")
            else:
                table_row.append(str(elm))

        table.add_row(*table_row, end_section=False)

    console = Console(record=True)
    console.print(table)


def plot_inference_report(report, with_baseline=False):
    # add title and labels
    device = report["device"].iloc[0]
    report["experiment_name"] = report.index

    # create bar charts seperately
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    fig2, ax2 = plt.subplots(figsize=(20, 10))

    axs = [ax1, ax2]

    sns.barplot(
        x="experiment_name",
        y="Model.Throughput(iter/s)",
        data=report,
        ax=ax1,
        width=0.5,
    )
    sns.barplot(
        x="experiment_name",
        y="Generation.Throughput(tok/s)",
        data=report,
        ax=ax2,
        width=0.5,
    )

    # add speedup text on top of each bar
    if with_baseline:
        baseline_throughput = report["Model.Throughput(iter/s)"].iloc[-1]
        for p in ax1.patches:
            speedup = (p.get_height() / baseline_throughput - 1) * 100
            ax1.annotate(
                f"{'+' if speedup>0 else '-'}{abs(speedup):.2f}%",
                (p.get_x() + p.get_width() / 2, 1.02 * p.get_height()),
                ha="center",
                va="center",
            )

        baseline_throughput = report["Generation.Throughput(tok/s)"].iloc[-1]
        for p in ax2.patches:
            speedup = (p.get_height() / baseline_throughput - 1) * 100
            ax2.annotate(
                f"{'+' if speedup>0 else '-'}{abs(speedup):.2f}%",
                (p.get_x() + p.get_width() / 2, 1.02 * p.get_height()),
                ha="center",
                va="center",
            )

    # add ticks
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment="right")

    # rename x axis
    ax1.set_xlabel("Experiment")
    ax2.set_xlabel("Experiment")

    # rename y axis
    ax1.set_ylabel("Forward Throughput (iter/s)")
    ax2.set_ylabel("Generate Throughput (tok/s)")

    axs[0].set_title(f"Model Throughput and Speedup on {device.upper()}")
    axs[1].set_title(f"Generation Throughput and Speedup on {device.upper()}")

    # save figures
    fig1.savefig(f"images/whisper_{device}_throughput.png", bbox_inches="tight")
    fig2.savefig(f"images/whisper_{device}_gen_throughput.png", bbox_inches="tight")

    # plt.show()


def main(args):
    # gather experiments reports
    experiments_dfs = []
    for experiment_folder in args.experiments:
        experiments_dfs.append(gather_inference_report(experiment_folder))

    if args.baseline:
        baseline_df = gather_inference_report(args.baseline)
        report = pd.concat([*experiments_dfs, baseline_df], axis=0)
        report["Model.Speedup(%)"] = (
            report["Model.Throughput(iter/s)"]
            / report["Model.Throughput(iter/s)"].iloc[-1]
            - 1
        ) * 100
        if "Generation.Throughput(tok/s)" in report.columns:
            report["Generation.Speedup(%)"] = (
                report["Generation.Throughput(tok/s)"]
                / report["Generation.Throughput(tok/s)"].iloc[-1]
                - 1
            ) * 100
    else:
        report = pd.concat(experiments_dfs, axis=0, ignore_index=True)

    show_inference_report(report, with_baseline=args.baseline is not None)
    plot_inference_report(report, with_baseline=args.baseline is not None)
    report.to_csv(f"inference_report.csv", index=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiments",
        "-e",
        nargs="*",
        type=Path,
        default="sweeps/",
        help="The folder containing the results of experiments.",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        type=Path,
        help="The folders containing the results of baseline.",
    )
    args = parser.parse_args()
    main(args)
