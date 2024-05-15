import subprocess

import pandas as pd
from huggingface_hub import create_repo, upload_file

scrapping_script = """
git clone https://github.com/Weyaxi/scrape-open-llm-leaderboard.git
pip install -r scrape-open-llm-leaderboard/requirements.txt
python scrape-open-llm-leaderboard/main.py
rm -rf scrape-open-llm-leaderboard
"""


def run_scrapper():
    subprocess.run(scrapping_script, shell=True)


def main():
    run_scrapper()

    open_llm_leaderboard = pd.read_csv("open-llm-leaderboard.csv")

    if len(open_llm_leaderboard) > 0:
        create_repo(repo_id="optimum-benchmark/open-llm-leaderboard", repo_type="dataset", exist_ok=True, private=False)
        upload_file(
            repo_id="optimum-benchmark/open-llm-leaderboard",
            commit_message="Update open LLM leaderboard",
            path_or_fileobj="open-llm-leaderboard.csv",
            path_in_repo="open-llm-leaderboard.csv",
            repo_type="dataset",
        )
    else:
        raise ValueError("No models found")


if __name__ == "__main__":
    main()
