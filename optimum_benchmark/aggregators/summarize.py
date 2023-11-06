from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional
import json

import pandas as pd


def summarize(
    report: pd.DataFrame,
    rename_dict: Dict[str, str],
):
    summarized_report = report[list(rename_dict.keys())].rename(columns=rename_dict)

    return summarized_report


def summarize_cli() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        default=Path("artifacts/full_report.csv"),
        help="Path to the full report csv file",
    )
    parser.add_argument(
        "--rename-json",
        "-j",
        type=Path,
        help="Path to the rename json file",
    )
    parser.add_argument(
        "--save-file",
        "-s",
        type=Path,
        default=Path("artifacts/summarized_report.csv"),
        help="Path to the report file",
    )

    args = parser.parse_args()

    if args.rename_json is None:
        rename_dict = {
            "experiment_name": "Experiment",
            "benchmark.name": "Benchmark",
            "backend.name": "Backend",
        }
    else:
        rename_dict = json.load(args.rename_json.open("r"))

    report = pd.read_csv(args.report)
    summarized_report = summarize(
        report=report,
        rename_dict=rename_dict,
    )

    if args.save_file is not None:
        args.save_file.parent.mkdir(parents=True, exist_ok=True)
        summarized_report.to_csv(args.save_file, index=False)
