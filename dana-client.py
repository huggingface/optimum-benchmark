from typing import Any, Dict, Optional
from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
import requests
import random
import json


def add_new_optimum_build(
        project_id: str,
        build_id: int,
        dashboard_url: str,
        override: bool = False,
        dry_run: bool = False,
        verbose: bool = False) -> None:
    """
        Posts a new build to the dashboard.
    """

    payload = {
        "projectId": project_id,
        "build": {
            "buildId": build_id,
            "infos": {
                "hash": "commit_hash",
                "abbrevHash": "commit_abbrev_hash",
                "authorName": "ilyas",
                "authorEmail": "ilyas@gmail.com",
                "subject": "commit_subject",
                "url": "commit_url",
            },
        },
        "override": override,
    }

    post_to_dashboard(f"{dashboard_url}/apis/addBuild",
                      payload,
                      dry_run=dry_run,
                      verbose=verbose)


def add_new_optimum_series(
    project_id: str,
    series_id: str,
    dashboard_url: str,
    series_description: Optional[str] = None,
    override: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    average_range: int = 5,
    average_min_count: int = 3,
    better_criterion: str = "lower"
) -> None:
    """
        Posts a new series to the dashboard.
    """

    payload = {
        "projectId": project_id,
        "serieId": series_id,
        "analyse": {
            "benchmark": {
                "range": average_range,
                "required": average_min_count,
                "trend": better_criterion,
            }
        },
        "override": override,
    }

    if series_description is not None:
        payload["description"] = series_description

    post_to_dashboard(f"{dashboard_url}/apis/addSerie",
                      payload,
                      dry_run=dry_run,
                      verbose=verbose)


def add_new_sample(
    project_id: str,
    series_id: str,
    build_id: int,
    sample_value: int,
    dashboard_url: str,
    override: bool = False,
    dry_run: bool = False,
    verbose: bool = False
) -> None:
    """
        Posts a new sample to the dashboard.
    """

    payload = {
        "projectId": project_id,
        "serieId": series_id,
        "sample": {
            "buildId": build_id,
            "value": sample_value
        },
        "override": override
    }

    post_to_dashboard(
        f"{dashboard_url}/apis/addSample",
        payload,
        dry_run=dry_run,
        verbose=verbose
    )


def post_to_dashboard(
    dashboard_url: str,
    payload: Dict[str, Any],
    dry_run: bool = False,
    verbose: bool = False,
) -> None:

    data = json.dumps(payload)

    if dry_run or verbose:
        print(f"API request payload: {data}")

    if dry_run:
        return

    response = requests.post(
        dashboard_url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": 'Bearer secret',
        }
    )
    code = response.status_code
    print(f"API response status code: {code}")


def main():
    DASHBOARD_URL = "http://localhost:7000"
    PROJECT_ID = "Optimum-Benchmark"
    BENCHMARKS_FOLDER = Path("runs")
    DRY_RUN = False
    VERBOSE = False

    build_id = random.randint(0, 1000000)

    print("Adding new build to dashboard...")
    add_new_optimum_build(
        project_id=PROJECT_ID,
        build_id=build_id,
        dashboard_url=DASHBOARD_URL,
        override=True,
        dry_run=DRY_RUN,
        verbose=VERBOSE,
    )

    for benchmark_foler in BENCHMARKS_FOLDER.iterdir():

        last_experiment = list(benchmark_foler.iterdir())[-1]
        series_id = 'serie' + '.' + benchmark_foler.name + "." + last_experiment.name

        results = pd.read_csv(
            last_experiment / "inference_results.csv")

        sample_value = results["Model latency mean (s)"][0]

        config = OmegaConf.load(last_experiment / ".hydra/config.yaml")
        series_description = OmegaConf.to_yaml(
            config)

        print("Adding new series to dashboard...")
        add_new_optimum_series(
            project_id=PROJECT_ID,
            series_id=series_id,
            series_description=series_description,
            dashboard_url=DASHBOARD_URL,
            override=True,
            dry_run=DRY_RUN,
            verbose=VERBOSE,
        )

        print("Adding new sample to dashboard...")
        add_new_sample(
            project_id=PROJECT_ID,
            series_id=series_id,
            build_id=build_id,
            sample_value=sample_value,
            dashboard_url=DASHBOARD_URL,
            override=True,
            dry_run=DRY_RUN,
            verbose=VERBOSE,
        )


if __name__ == "__main__":
    main()
