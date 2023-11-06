import sys

from .aggregators.gather import gather_cli
from .aggregators.display import display_cli
from .aggregators.summarize import summarize_cli
from .aggregators.plot import plot_cli

HELP = """
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


def main():
    action = sys.argv[1]
    sys.argv = sys.argv[1:]

    if action == "gather":
        gather_cli()
    elif action == "display":
        display_cli()
    elif action == "summarize":
        summarize_cli()
    elif action == "plot":
        plot_cli()
    elif action in ["-h", "--help"]:
        print(HELP)
    else:
        raise ValueError(f"Unknown action {action}")
