import glob
import os
import sys
from logging import getLogger

import hydra
from omegaconf import DictConfig

from .experiment import run_with_launcher

LOGGER = getLogger("main")


@hydra.main(version_base=None)
# hydra takes care of the cli and returns the config object
def benchmark_cli(experiment: DictConfig) -> None:
    if glob.glob("*.csv") and os.environ.get("OVERRIDE_BENCHMARKS", "0") != "1":
        LOGGER.warning(
            "Skipping benchmark because results already exist. "
            "Set OVERRIDE_BENCHMARKS=1 to override benchmark results."
        )
        return

    run_with_launcher(experiment)


def report_cli() -> None:
    action = sys.argv[1]
    sys.argv = sys.argv[1:]

    if action == "gather":
        from .aggregators.gather import gather_cli

        gather_cli()
    elif action == "display":
        from .aggregators.display import display_cli

        display_cli()
    elif action == "summarize":
        from .aggregators.summarize import summarize_cli

        summarize_cli()
    elif action == "plot":
        from .aggregators.plot import plot_cli

        plot_cli()
    elif action in ["-h", "--help"]:
        print(
            """
            Usage: optimum-report <action> <options>
                Actions:
                    gather
                    display
                    summarize
                    plot
                    -h, --help

            For more information on each action, run:
                optimum-report <action> -h
            """
        )
    else:
        raise ValueError(f"Unknown action {action}")
