import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from flatten_dict import flatten
from argparse import ArgumentParser

from rich.console import Console
from rich.table import Table


def gather_inference_report(
    folder: Path,
) -> pd.DataFrame:
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
    inference_report["Config Path"] = configs_files  # for console display (clickable)
    inference_report.set_index("Config Path", inplace=True)
    # sort by throughput and remove failed experiments
    inference_report.sort_values(
        by=["Model.Throughput(iter/s)"], ascending=False, inplace=True
    )

    return inference_report


def show_inference_report(report, with_baseline=False):
    # columns to display
    show_report = report[
        (["Baseline"] if with_baseline else [])
        + ["Model.Latency(s)", "Model.Throughput(iter/s)"]
        + (
            ["Model.Peak_Memory(MB)"]
            if "Model.Peak_Memory(MB)" in report.columns
            else []
        )
        + (["Speedup(%)"] if with_baseline else [])
        + (
            ["Generation.Throughput(tok/s)"]
            if "Generation.Throughput(tok/s)" in report.columns
            else []
        )
    ]

    table = Table(
        show_header=True,
        title="Inference Benchmark Report",
    )

    show_report.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split(".")) for col in show_report.columns.to_list()]
    )

    table.add_column("Config Path", justify="left")
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
                else:
                    table_row.append(f"{elm}")

            elif type(elm) == bool:
                if elm:
                    table_row.append("[green]✔[/green]")
                else:
                    table_row.append("[red]✘[/red]")
            elif elm is None:
                table_row.append(None)
            else:
                table_row.append(str(elm))

        table.add_row(*table_row, end_section=False)

    console = Console(record=True)
    console.print(table)


def main(args):
    experiments_report = gather_inference_report(args.experiments_folder)
    if args.baseline_folder is not None:
        print("Using the provided baseline")
        baseline_report = gather_inference_report(args.baseline_folder)
        assert len(baseline_report) == 1, "There should be only one baseline"
        experiments_report["Baseline"] = [False] * len(experiments_report)
        experiments_report["Speedup(%)"] = (
            experiments_report["Model.Throughput(iter/s)"]
            / baseline_report["Model.Throughput(iter/s)"].iloc[0]
            - 1
        ) * 100
        baseline_report["Baseline"] = [True] * len(baseline_report)
        baseline_report["Speedup(%)"] = [0] * len(baseline_report)

        report = pd.concat(
            [experiments_report, baseline_report], axis=0, ignore_index=False
        )
    else:
        print("No baseline provided")
        report = experiments_report

    report.to_csv(f"{args.experiments_folder}/inference_report.csv", index=True)
    show_inference_report(report, with_baseline=args.baseline_folder is not None)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiments-folder",
        "-e",
        type=Path,
        default="sweeps/",
        help="The folder containing the results of experiments.",
    )
    parser.add_argument(
        "--baseline-folder",
        "-b",
        type=Path,
        help="The folder containing the results of the baseline.",
    )
    args = parser.parse_args()
    main(args)
