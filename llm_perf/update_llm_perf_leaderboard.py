import subprocess
from glob import glob
from tempfile import TemporaryDirectory

import pandas as pd
from huggingface_hub import create_repo, snapshot_download, upload_file
from tqdm import tqdm

from optimum_benchmark import Benchmark


def gather_benchmarks(subset: str, machine: str):
    pull_repo_id = f"optimum-benchmark/llm-perf-pytorch-cuda-{subset}-{machine}"
    snapshot = snapshot_download(repo_type="dataset", repo_id=pull_repo_id, allow_patterns=["**/benchmark.json"])

    dfs = []
    for file in tqdm(glob(f"{snapshot}/**/benchmark.json", recursive=True)):
        dfs.append(Benchmark.from_json(file).to_dataframe())
    benchmarks = pd.concat(dfs, ignore_index=True)

    tmp_dir = TemporaryDirectory()
    push_repo_id = "optimum-benchmark/llm-perf-leaderboard"
    file_name = f"llm-perf-leaderboard-{subset}-{machine}.csv"
    benchmarks.to_csv(f"{tmp_dir.name}/{file_name}", index=False)

    create_repo(repo_id=push_repo_id, repo_type="dataset", private=False, exist_ok=True)
    upload_file(
        path_or_fileobj=f"{tmp_dir.name}/{file_name}", path_in_repo=file_name, repo_id=push_repo_id, repo_type="dataset"
    )
    tmp_dir.cleanup()


def update_perf_dfs():
    for subset in ["unquantized", "bnb", "awq", "gptq"]:
        for machine in ["1xA10", "1xA100"]:
            try:
                gather_benchmarks(subset, machine)
            except Exception:
                print(f"Subset {subset} for machine {machine} not found")


scrapping_script = """
git clone https://github.com/Weyaxi/scrape-open-llm-leaderboard.git
pip install -r scrape-open-llm-leaderboard/requirements.txt
python scrape-open-llm-leaderboard/main.py
rm -rf scrape-open-llm-leaderboard
"""


def update_llm_df():
    subprocess.run(scrapping_script, shell=True)
    create_repo(repo_id="optimum-benchmark/open-llm-leaderboard", repo_type="dataset", exist_ok=True, private=False)
    upload_file(
        repo_id="optimum-benchmark/open-llm-leaderboard",
        commit_message="Update open LLM leaderboard",
        path_or_fileobj="open-llm-leaderboard.csv",
        path_in_repo="open-llm-leaderboard.csv",
        repo_type="dataset",
    )


if __name__ == "__main__":
    update_llm_df()
    update_perf_dfs()
