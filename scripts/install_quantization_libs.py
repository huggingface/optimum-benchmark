import argparse
import os
import subprocess
import sys

EXTERNAL_REPOS_DIR = "external_repos"


def clone_or_pull_repo(repo_url, repo_location_path):
    """Clone the repo if it doesn't exist; otherwise, pull the latest changes."""
    if os.path.exists(repo_location_path):
        print(f"Directory {repo_location_path} already exists. Pulling the latest changes.")
        subprocess.run(f"cd {repo_location_path} && git pull", shell=True, check=True)
    else:
        repo_name = repo_location_path.split("/")[-1]
        print(f"Cloning {repo_name} into {repo_location_path}")
        subprocess.run(f"git clone {repo_url} {repo_location_path}", shell=True, check=True)


def install_autoawq_from_source():
    """Install the AutoAWQ and AutoAWQ_kernels packages from GitHub."""
    print("Installing AutoAWQ and AutoAWQ_kernels packages.")
    autoawq_repo_name = "AutoAWQ"
    autoawq_kernels_repo_name = "AutoAWQ_kernels"

    autoawq_repo_path = os.path.join(EXTERNAL_REPOS_DIR, autoawq_repo_name)
    kernels_repo_path = os.path.join(EXTERNAL_REPOS_DIR, autoawq_kernels_repo_name)

    print("Installing AutoAWQ_kernels package.")
    clone_or_pull_repo(f"https://github.com/casper-hansen/{autoawq_kernels_repo_name}", kernels_repo_path)
    subprocess.run(
        f"cd {kernels_repo_path} && {sys.executable} -m pip install . --no-build-isolation",
        shell=True,
        check=True,
        env=os.environ,
    )

    print("Installing AutoAWQ package.")
    clone_or_pull_repo(f"https://github.com/casper-hansen/{autoawq_repo_name}", autoawq_repo_path)
    subprocess.run(
        f"cd {autoawq_repo_path} && {sys.executable} -m pip install . --no-build-isolation",
        shell=True,
        check=True,
        env=os.environ,
    )

    print("AutoAWQ and AutoAWQ_kernels packages installed.")


def install_gptqmodel_from_source():
    """Install the GPTQModel package from GitHub."""
    print("Installing GPTQModel package.")
    gptqmodel_repo_path = os.path.join(EXTERNAL_REPOS_DIR, "GPTQModel")

    clone_or_pull_repo("https://github.com/ModelCloud/GPTQModel.git", gptqmodel_repo_path)
    subprocess.run(
        f"cd {gptqmodel_repo_path} && {sys.executable} -m pip install . --no-build-isolation",
        shell=True,
        check=True,
        env=os.environ,
    )

    print("GPTQModel package installed.")


def main():
    parser = argparse.ArgumentParser(description="Install AutoAWQ and/or GPTQModel packages from source.")
    parser.add_argument(
        "--autoawq",
        action="store_true",
        help="Install AutoAWQ and AutoAWQ_kernels packages from source.",
    )
    parser.add_argument(
        "--gptqmodel",
        action="store_true",
        help="Install GPTQModel package from source.",
    )

    args = parser.parse_args()

    if args.autoawq:
        install_autoawq_from_source()
    if args.gptqmodel:
        install_gptqmodel_from_source()

    if not args.autoawq and not args.gptqmodel:
        print("Please specify an installation option. Use --autoawq or --gptqmodel.")
        sys.exit(1)


if __name__ == "__main__":
    main()
