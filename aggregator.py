import pandas as pd
from json import dumps
from typing import Tuple
from pathlib import Path
from omegaconf import OmegaConf
from flatten_dict import flatten
from argparse import ArgumentParser
from flatten_dict.reducers import make_reducer

from rich.console import Console
from rich.table import Table


def gather_results(folder: Path) -> Tuple[dict, pd.DataFrame]:
    """
    Gather all results and configs from the given folder.

    Parameters
    ----------
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
    stats_f = [f for f in folder.glob("**/inference_results.csv")]
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

    # remove unnecessary columns
    report.drop(
        columns=[col for col in report.columns if "_target_" in col], inplace=True
    )

    # extract static parameters
    static_params = (
        report.loc[:, report.nunique() == 1]
        .fillna(method="bfill")
        .drop_duplicates()
        .to_dict(orient="records")[0]
    )

    # extract benchmark parameters
    bench_results = report.loc[:, report.nunique() > 1].sort_values(
        by=[col for col in report.columns if "latency mean" in col], ascending=True
    )

    bench_results = bench_results[
        [col for col in bench_results.columns if "Model" not in col]
        + [col for col in bench_results.columns if "Model" in col]
    ]

    return static_params, bench_results


def show_results_in_console(static_params: dict, bench_results: pd.DataFrame) -> None:
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
        header_style="bold",
        title="Stats per run",
    )

    # Define the columns
    for col in bench_results.columns:
        table.add_column(col)

    # Add rows
    for row in bench_results.itertuples(index=False):
        # stringify the row
        table.add_row(*[str(v) for v in row])

    console = Console()
    # Display environment
    console.print("Static Parameters:", static_params)
    # Display the table
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

    static_params, bench_results = gather_results(args.folder)

    # Save aggregated results
    with open(f"{args.folder}/static_params.json", "w") as f:
        f.write(dumps(static_params, indent=4))
    bench_results.to_csv(f"{args.folder}/bench_results.csv", index=False)

    # Display the results
    show_results_in_console(static_params, bench_results)
