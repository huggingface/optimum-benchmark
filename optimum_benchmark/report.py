import sys

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
        print(HELP)
    else:
        raise ValueError(f"Unknown action {action}")
