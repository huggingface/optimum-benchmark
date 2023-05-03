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
    environment : dict
        The environment configuration represented by parameters that are the same for all runs
    report : pd.DataFrame
        The report containing all stats with their corresponding configurations in a single dataframe.
    """

    # List all csv results
    stats_f = [f for f in folder.glob("**/stats.json")]
    configs_f = [f for f in folder.glob("**/config.yaml")]

    stats_dfs = {
        f.relative_to(folder).parent.as_posix(): pd.read_json(f, orient='index').transpose()
        for f in stats_f
    }

    configs_dfs = {
        f.relative_to(folder).parent.parent.as_posix(): pd.DataFrame(
            flatten(
                OmegaConf.load(f),
                reducer=make_reducer(delimiter='.')
            ), index=[0]
        )
        for f in configs_f
    }

    if len(stats_dfs) == 0 or len(configs_dfs) == 0:
        raise ValueError(f"No results found in {folder}")

    # Merge perfs dataframes with configs
    reports = {
        name: configs_dfs[name].merge(
            stats_dfs[name], left_index=True, right_index=True)
        for name in stats_dfs.keys()
    }

    report = pd.concat(reports.values(), axis=0, ignore_index=True)

    # remove unnecessary columns
    report.drop(
        columns=[col for col in report.columns if '_target_' in col],
        inplace=True
    )
    environment = report.loc[:, report.nunique(
    ) == 1].drop_duplicates().to_dict(orient='records')[0]
    report = report.loc[:, report.nunique() > 1]

    return environment, report


def show_results_in_console(environement: dict, report: pd.DataFrame) -> None:
    """
    Display the results in the console.

    Parameters
    ----------
    environement : dict
        The environment configuration represented by parameters that are the same for all runs
    report : pd.DataFrame
        The report containing all stats with their corresponding configurations in a single dataframe.
    """

    table = Table(
        show_header=True, header_style="bold",
        title="Stats per run",
    )

    # Define the columns
    for col in report.columns:
        table.add_column(col)

    # Add rows
    for row in report.itertuples(index=False):
        # stringify the row
        table.add_row(*[str(v) for v in row])

    console = Console()
    # Display environment
    console.print('ENV:', environement)
    # Display the table
    console.print(table)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--folder', '-f', type=Path, default='sweeps/',
        help="The folder containing the results of the benchmark."
    )
    args = parser.parse_args()

    environment, report = gather_results(args.folder)

    # Save aggregated results
    with open(f'{args.folder}/environment.json', 'w') as f:
        f.write(dumps(environment, indent=4))
    report.to_csv(f'{args.folder}/report.csv', index=False)

    # Display the results
    show_results_in_console(environment, report)
