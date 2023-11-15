from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
from flatten_dict import flatten
from omegaconf import OmegaConf


def gather(root_folders: List[Path]) -> pd.DataFrame:
    results_dfs = {}
    configs_dfs = {}

    for root_folder in root_folders:
        if not root_folder.exists():
            raise ValueError(f"{root_folder} does not exist")

        for f in root_folder.glob("**/*_results.csv"):
            parent_folder = f.parent.absolute().as_posix()
            results_df = pd.read_csv(f)
            results_dfs[parent_folder] = results_df

        for f in root_folder.glob("**/hydra_config.yaml"):
            parent_folder = f.parent.absolute().as_posix()
            config_df = pd.DataFrame.from_dict(flatten(OmegaConf.load(f), reducer="dot"), orient="index").T
            configs_dfs[parent_folder] = config_df

    if (len(results_dfs) == 0) or (len(configs_dfs) == 0):
        raise ValueError(f"No results found in {root_folder}")

    # Merge inference and config dataframes
    full_dfs = {}
    for parent_folder in results_dfs:
        full_df = pd.merge(results_dfs[parent_folder], configs_dfs[parent_folder], left_index=True, right_index=True)
        full_dfs[parent_folder] = full_df

    # Concatenate all dataframes
    full_report = pd.concat(full_dfs.values(), ignore_index=True, axis=0)

    return full_report


def gather_cli() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--root-folders",
        "-r",
        nargs="+",
        type=Path,
        required=True,
        help="The folders containing the experiments to report on.",
    )
    parser.add_argument(
        "--save-file",
        "-s",
        type=Path,
        default=Path("artifacts/full_report.csv"),
        help="Path to the report file",
    )

    args = parser.parse_args()
    report = gather(root_folders=args.root_folders)

    if args.save_file is not None:
        args.save_file.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.save_file, index=False)
