import os
from glob import glob
from tempfile import TemporaryDirectory

import pandas as pd
from huggingface_hub import create_repo, snapshot_download, upload_file
from tqdm import tqdm

from optimum_benchmark import Benchmark

SUBSET = os.getenv("SUBSET")
MACHINE = os.getenv("MACHINE")

PULL_REPO_ID = f"optimum-benchmark/llm-perf-pytorch-cuda-{SUBSET}-{MACHINE}"

print("Pulling benchmark data from", PULL_REPO_ID)
snapshot = snapshot_download(
    repo_id=PULL_REPO_ID,
    repo_type="dataset",
    allow_patterns=["**/benchmark.json"],
)


def gather_benchmarks():
    dfs = []
    for file in tqdm(glob(f"{snapshot}/**/benchmark.json", recursive=True)):
        dfs.append(Benchmark.from_json(file).to_dataframe())

    benchmarks = pd.concat(dfs, ignore_index=True)
    return benchmarks


benchmarks = gather_benchmarks()

with TemporaryDirectory() as tmp_dir:
    PUSH_REPO_ID = "optimum-benchmark/llm-perf-leaderboard"
    FILE_NAME = f"llm-perf-leaderboard-{SUBSET}-{MACHINE}.csv"

    benchmarks.to_csv(f"{tmp_dir}/{FILE_NAME}", index=False)

    create_repo(repo_id=PUSH_REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    upload_file(
        path_or_fileobj=f"{tmp_dir}/{FILE_NAME}",
        path_in_repo=FILE_NAME,
        repo_id=PUSH_REPO_ID,
        repo_type="dataset",
    )
