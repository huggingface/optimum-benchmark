from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
from rich.table import Table
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from rich.console import Console
from flatten_dict import flatten
from rich.terminal_theme import MONOKAI


def gather(root_folders: List[Path]) -> pd.DataFrame:
    configs_dfs = {}
    results_dfs = {}

    for root_folder in root_folders:
        if not root_folder.exists():
            raise ValueError(f"{root_folder} does not exist")

        for f in root_folder.glob("**/hydra_config.yaml"):
            parent_folder = f.parent.absolute().as_posix()
            configs_dfs[parent_folder] = pd.DataFrame.from_dict(
                flatten(OmegaConf.load(f), reducer="dot"), orient="index"
            ).T

        for f in root_folder.glob("**/*_results.csv"):
            parent_folder = f.parent.absolute().as_posix()
            results_dfs[parent_folder] = pd.read_csv(f)

    if (len(results_dfs) == 0) or (len(configs_dfs) == 0):
        raise ValueError(f"Results are missing in {root_folders}")

    # Merge inference and config dataframes
    full_dfs = {}
    for parent_folder in results_dfs:
        full_df = pd.concat(
            [configs_dfs[parent_folder], results_dfs[parent_folder]],
            axis=1,
        )
        full_df["parent_folder"] = parent_folder
        full_dfs[parent_folder] = full_df

    # Concatenate all dataframes
    full_report = pd.concat(full_dfs.values(), ignore_index=True, axis=0)

    return full_report


def format_element(element):
    if isinstance(element, float):
        if element != element:
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
            formated_element = "[green]✔[/green]"
        else:
            formated_element = "[red]✘[/red]"
    else:
        formated_element = str(element)

    return formated_element


def display(report: pd.DataFrame) -> Table:
    table = Table(show_header=True, show_lines=True)

    for column in report.columns:
        table.add_column(column, justify="right", header_style="bold")

    for _, row in report.iterrows():
        formated_row = []
        for element in row.values:
            formated_row.append(format_element(element))
        table.add_row(*formated_row)

    console = Console(record=True, theme=MONOKAI)
    console.print(table, justify="center")

    return console, table


def rename(report: pd.DataFrame, rename_dict: Dict[str, str]):
    summarized_report = report[list(rename_dict.keys())].rename(columns=rename_dict)

    return summarized_report


def plot(report: pd.DataFrame, x_axis: str, y_axis: str, groupby: str) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()

    for group, sweep in report.groupby(groupby):
        sorted_sweep = sweep.sort_values(by=x_axis)
        ax.plot(sorted_sweep[x_axis], sorted_sweep[y_axis], label=group, marker="o")

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{y_axis} per {x_axis}")
    ax.legend(fancybox=True, shadow=True)

    return fig, ax
