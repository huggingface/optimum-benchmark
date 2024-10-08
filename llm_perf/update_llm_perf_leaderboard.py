import subprocess
from glob import glob

import pandas as pd
from huggingface_hub import create_repo, snapshot_download, upload_file
from tqdm import tqdm

from optimum_benchmark import Benchmark

REPO_TYPE = "dataset"
MAIN_REPO_ID = "optimum-benchmark/llm-perf-leaderboard"
PERF_REPO_ID = "optimum-benchmark/llm-perf-{backend}-{hardware}-{subset}-{machine}"

PERF_DF = "perf-df-{subset}-{machine}.csv"
LLM_DF = "llm-df.csv"


def gather_benchmarks(subset: str, machine: str, backend: str, hardware: str):
    """
    Gather the benchmarks for a given machine
    """
    perf_repo_id = PERF_REPO_ID.format(subset=subset, machine=machine, backend=backend, hardware=hardware)
    snapshot = snapshot_download(repo_type=REPO_TYPE, repo_id=perf_repo_id, allow_patterns=["**/benchmark.json"])

    dfs = []
    for file in tqdm(glob(f"{snapshot}/**/benchmark.json", recursive=True)):
        dfs.append(Benchmark.from_json(file).to_dataframe())
    benchmarks = pd.concat(dfs, ignore_index=True)

    perf_df = PERF_DF.format(subset=subset, machine=machine)
    benchmarks.to_csv(perf_df, index=False)
    create_repo(repo_id=MAIN_REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)
    upload_file(repo_id=MAIN_REPO_ID, repo_type=REPO_TYPE, path_in_repo=perf_df, path_or_fileobj=perf_df)


def update_perf_dfs():
    """
    Update the performance dataframes for all machines
    """
    for machine in ["1xA10", "1xA100", "1xT4", "32vCPU-C7i"]:
        for backend in ["pytorch"]:
            for hardware in ["cuda", "cpu"]:
                for subset in ["unquantized", "bnb", "awq", "gptq"]:
                    try:
                        gather_benchmarks(subset, machine, backend, hardware)
                    except Exception:
                        print(
                            f"benchmark for subset: {subset}, machine: {machine}, backend: {backend}, hardware: {hardware} not found"
                        )


scrapping_script = """
git clone https://github.com/Weyaxi/scrape-open-llm-leaderboard.git
pip install -r scrape-open-llm-leaderboard/requirements.txt
python scrape-open-llm-leaderboard/main.py
rm -rf scrape-open-llm-leaderboard
"""


def update_llm_df():
    """
    Scrape the open-llm-leaderboard and update the leaderboard dataframe
    """
    subprocess.run(scrapping_script, shell=True)
    create_repo(repo_id=MAIN_REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)
    upload_file(
        repo_id=MAIN_REPO_ID, repo_type=REPO_TYPE, path_in_repo=LLM_DF, path_or_fileobj="open-llm-leaderboard.csv"
    )


if __name__ == "__main__":
    update_llm_df()
    update_perf_dfs()
