from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.terminal_theme import MONOKAI


def style_element(element, style=""):
    if style:
        return f"[{style}]{element}[/{style}]"
    else:
        return element


def format_element(element, style=""):
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


def display(report: pd.DataFrame) -> Table:
    table = Table(show_header=True, show_lines=True)

    for column in report.columns:
        table.add_column(column, justify="right", header_style="bold")

    for _, row in report.iterrows():
        table.add_row(*format_row(row.values, style=""))

    console = Console(record=True)
    console.print(table, justify="center")

    return console, table


def display_cli() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        default=Path("artifacts/short_report.csv"),
        help="The short report to format.",
    )
    parser.add_argument(
        "--save-file",
        "-s",
        type=Path,
        default=Path("artifacts/rich_table.svg"),
        help="Path to the svg file",
    )

    args = parser.parse_args()
    report = pd.read_csv(args.report)
    console, _ = display(report=report)

    if args.save_file is not None:
        args.save_file.parent.mkdir(parents=True, exist_ok=True)
        console.save_svg(args.save_file, theme=MONOKAI, title="Rich Report")
