import torch
from typing import Dict
from transformers import AutoConfig

from src.benchmark.config import BenchmarkConfig

def get_input_ids(config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
    """Generate random input ids"""
    return torch.randint(
        low=0,
        high=AutoConfig.from_pretrained(config.model).vocab_size,
        size=(config.batch_size, config.sequence_length),
        dtype=torch.long,
        device=config.backend.device,
    )


def get_attention_mask(config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
    """Generate random attention mask"""
    mask = torch.ones(
        config.batch_size,
        config.sequence_length,
        dtype=torch.long,
        device=config.backend.device,
    )
    
    # masking out a certain ratio (config.sparsity) of tokens
    mask = mask * torch.distributions.Bernoulli(
        torch.tensor([config.sparsity], device=config.backend.device)
    ).sample((config.batch_size, config.sequence_length)).long()

    return mask


def get_token_ids(config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
    """Generate random token type ids"""
    return torch.ones(
        config.batch_size,
        config.sequence_length,
        dtype=torch.long,
        device=config.backend.device,
    )


INPUT_GENERATORS = {
    'input_ids': get_input_ids,
    'attention_mask': get_attention_mask,
    'token_type_ids': get_token_ids
}

from collections import defaultdict
from pathlib import Path

import pandas as pd
from argparse import ArgumentParser

# from rich.console import Console
# from rich.table import Table


def gather_sweep_results(folder: Path):
    # List all csv results
    results_f = [f for f in folder.glob("**/perfs.csv")]
    results_csv = {
        f.relative_to(folder).parent.as_posix(): pd.read_csv(f, index_col=0)
        for f in results_f
    }

    if len(results_csv) == 0:
        raise ValueError(f"No perfs.csv file were found in {folder}")

    # Merge dataframe wrt to framework
    dfs = defaultdict(list)
    for path, df in results_csv.items():
        framework, device, arguments = path.split("/")
        arguments = dict(arg.split("_") for arg in arguments.split("-"))

        # Add columns to the dataframe
        for col_name, col_value in arguments.items():
            df[col_name] = int(col_value)

        dfs[framework].append(df)

    # Concat the dataframes
    dfs = {f: pd.concat(a) for f, a in dfs.items()}

    for framework, df in dfs.items():
        df["framework"] = framework

    return pd.concat(dfs.values())


def show_results_in_console(df):
    grouped_df = df.groupby(["framework", "batch", "seqlen"])
    (grouped_df["inference_time_secs"].mean() * 1000).reset_index()

    console = Console()
    table = Table(
        show_header=True, header_style="bold",
        title="Inference Time per Framework, Batch Size & Sequence Length"
    )

    columns = (
        ("Framework", "framework"),
        ("Batch Size", "batch"),
        ("Seq Length", "seqlen"),
        ("Inference Time (ms)", "inference_time_secs")
    )

    # Define the columns
    for (column, _) in columns:
        table.add_column(column, justify="center")

    # Add rows
    for name, group in grouped_df:
        items = name + (round(group.mean()["inference_time_secs"] * 1000, 2), )
        table.add_row(*[str(item) for item in items])

    # Display the table
    console.print(table)


if __name__ == '__main__':
    parser = ArgumentParser("Hugging Face Model Benchmark")
    parser.add_argument("--results-folder", type=Path, help="Where the benchmark results have been saved")
    parser.add_argument("output_folder", type=Path, help="Where the resulting report will be saved")

    # Parse command line arguments
    args = parser.parse_args()

    if not args.results_folder.exists():
        print(f"Folder {args.results_folder} doesn't exist")

    try:
        # Ensure output folder exists
        args.output_folder.mkdir(exist_ok=True, parents=True)

        # Gather the results to manipulate
        df_by_framework = gather_results(args.results_folder)

        # Generate reports
        df_by_framework.to_csv(args.output_folder.joinpath("final_results.csv"))

        show_results_in_console(df_by_framework)
    except ValueError as ve:
        print(ve)