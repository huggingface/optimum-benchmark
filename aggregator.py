import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from flatten_dict import flatten
from argparse import ArgumentParser
from flatten_dict.reducers import make_reducer

from rich.console import Console
from rich.table import Table


def gather_results(
    benchamrk: str, folder: Path
) -> pd.DataFrame:  # Tuple[dict, pd.DataFrame]:
    """
    Gather all results and configs from the given folder.

    Parameters
    ----------
    benchamrk : str
        The type of the benchmarks.
    folder : Path
        The folder to search for results.

    Returns
    -------
    static_params : dict
        The unchanging parameters of the benchmark.
    bench_results : pd.DataFrame
        The results of the benchmark with their corresponding changing parameters.
    """

    # List all csv results
    stats_f = [f for f in folder.glob(f"**/{benchamrk}_results.csv")]
    # List all configs except the ones in hydra folder
    configs_f = [
        f for f in folder.glob("**/config.yaml") if "hydra" not in f.as_posix()
    ]

    stats_dfs = {
        f.relative_to(folder).parent.as_posix(): pd.read_csv(f, index_col=0)
        for f in stats_f
    }

    configs_dfs = {
        f.relative_to(folder).parent.as_posix(): pd.DataFrame(
            flatten(OmegaConf.load(f), reducer=make_reducer(delimiter=".")), index=[0]  # type: ignore
        )
        for f in configs_f
    }

    if len(stats_dfs) == 0 or len(configs_dfs) == 0:
        raise ValueError(f"No results found in {folder}")

    # Merge perfs dataframes with configs
    reports = {
        name: configs_dfs[name].merge(
            stats_dfs[name], left_index=True, right_index=True
        )
        for name in stats_dfs.keys()
    }

    # Concatenate all reports
    report = pd.concat(reports.values(), axis=0, ignore_index=True)
    # set experiment_id as index
    report.set_index("experiment_id", inplace=True)
    # remove unnecessary columns
    report.drop(
        columns=[
            col
            for col in report.columns
            if ("_target_" in col)
            or ("version" in col)
            or (col in ["experiment_name", "task"])
        ],
        inplace=True,
    )

    return report


def show_results_in_console(report) -> None:
    """
    Display the results in the console.

    Parameters
    ----------
    static_params : dict
        The unchanging parameters of the benchmark.
    bench_results : pd.DataFrame
        The results of the benchmark with their corresponding changing parameters.
    """

    table = Table(
        show_header=True,
        padding=(0, 0),
        title="Benchmark Results",
    )

    report.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split(".")) for col in report.columns]
    )

    for level in range(report.columns.nlevels):
        if level == 0:
            pass
            columns = report.columns.get_level_values(level).to_list()
            for col in columns:
                table.add_column(col)

        else:
            columns = report.columns.get_level_values(level).to_list()

            for i in range(len(columns)):
                # if it's nan we don't want to print it
                if columns[i] != columns[i]:
                    # white background
                    columns[i] = ""

            table.add_row(*columns, end_section=True)

    for row in report.itertuples(index=False):
        table_row = []
        for elm in row:
            if type(elm) == float:
                table_row.append(f"{elm:.2e}")
            elif type(elm) == bool:
                if elm:
                    table_row.append("[green]✔[/green]")
                else:
                    table_row.append("[red]✘[/red]")
            elif elm is None:
                table_row.append("[yellow]N/A[/yellow]")
            else:
                table_row.append(str(elm))

        table.add_row(*table_row)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--folder",
        "-f",
        type=Path,
        default="sweeps/",
        help="The folder containing the results of the benchmark.",
    )
    args = parser.parse_args()
    report = gather_results("inference", args.folder)
    report.to_csv(f"{args.folder}/report.csv", index=False)

    # Display the results
    show_results_in_console(report)
