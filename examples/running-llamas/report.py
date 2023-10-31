from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from flatten_dict import flatten
from omegaconf import OmegaConf
from pandas import DataFrame
from rich.console import Console
from rich.table import Table
from rich.terminal_theme import MONOKAI


def gather_inference_report(root_folder: Path) -> DataFrame:
    # key is path to inference file as string, value is dataframe
    inference_dfs = {
        f.parent.absolute().as_posix(): pd.read_csv(f) for f in root_folder.glob("**/inference_results.csv")
    }

    # key is path to config file as string, value is flattened dict
    config_dfs = {
        f.parent.absolute()
        .as_posix(): pd.DataFrame.from_dict(flatten(OmegaConf.load(f), reducer="dot"), orient="index")
        .T
        for f in root_folder.glob("**/hydra_config.yaml")
        if f.parent.absolute().as_posix() in inference_dfs.keys()
    }

    if len(inference_dfs) == 0 or len(config_dfs) == 0:
        raise ValueError(f"No results found in {root_folder}")

    # Merge inference and config dataframes
    inference_reports = [
        config_dfs[name].merge(inference_dfs[name], left_index=True, right_index=True) for name in inference_dfs.keys()
    ]

    # Concatenate all reports
    inference_report = pd.concat(inference_reports, axis=0, ignore_index=True)
    inference_report.set_index("experiment_name", inplace=True)
    return inference_report


def style_element(element, style=""):
    if style:
        return f"[{style}]{element}[/{style}]"
    else:
        return element


def format_element(element, style=""):
    if isinstance(element, float):
        if element != element:  # nan
            formated_element = ""
        elif abs(element) >= 1:
            formated_element = f"{element:.2f}"
        elif abs(element) > 1e-6:
            formated_element = f"{element:.2e}"
        else:
            formated_element = f"{element}"
    elif element is None:
        formated_element = ""
    elif isinstance(element, bool):
        if element:
            formated_element = style_element("✔", style="green")
        else:
            formated_element = style_element("✘", style="red")
    else:
        formated_element = str(element)

    return style_element(formated_element, style=style)


def format_row(row, style=""):
    formated_row = []
    for element in row:
        formated_row.append(format_element(element, style=style))
    return formated_row


def get_short_report(inference_report):
    short_columns = {
        "environment.gpus": "GPU",
        "benchmark.input_shapes.batch_size": "Batch Size",
        "forward.latency(s)": "Forward Latency (s)",
        "forward.throughput(samples/s)": "Forward Throughput (samples/s)",
        "forward.max_memory_used(MB)": "Forward Max Memory Used (MB)",
        "forward.max_memory_allocated(MB)": "Forward Max Memory Allocated (MB)",
        "forward.max_memory_reserved(MB)": "Forward Max Memory Reserved (MB)",
        "generate.throughput(tokens/s)": "Generate Throughput (tokens/s)",
        "generate.max_memory_used(MB)": "Generate Max Memory Used (MB)",
        "generate.max_memory_allocated(MB)": "Generate Max Memory Allocated (MB)",
        "generate.max_memory_reserved(MB)": "Generate Max Memory Reserved (MB)",
    }
    short_report = inference_report[list(short_columns.keys())].rename(columns=short_columns)
    short_report["Quantization Scheme"] = inference_report.index.str.split("-").str[0]
    short_report["Quantization Scheme"].fillna("unquantized", inplace=True)
    short_report["Quantization Scheme"].replace("bnb", "BnB", inplace=True)
    short_report["Quantization Scheme"].replace("gptq", "GPTQ", inplace=True)

    short_report["GPU"] = short_report["GPU"].str[0]
    short_report["GPU"].replace("AMD INSTINCT MI250 (MCM) OAM AC MBA", "MI250", inplace=True)
    short_report["GPU"].replace("NVIDIA A100-SXM4-80GB", "A100", inplace=True)

    short_report["Group"] = short_report["GPU"] + "-" + short_report["Quantization Scheme"]

    return short_report


def get_rich_table(short_report):
    # create rich table
    rich_table = Table(show_header=True, show_lines=True)
    # we add a column for the index
    rich_table.add_column("Experiment Name", justify="left", header_style="")
    # we populate the table with values
    for column in short_report.columns:
        rich_table.add_column(column, justify="right", header_style="bold")
    # we add rows
    for index, row in short_report.iterrows():
        rich_table.add_row(index, *format_row(row.values, style=""))

    return rich_table


def get_throughput_plot(short_report):
    # for each quantization scheme we plot the throughput vs batch size
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    for group in short_report["Group"].unique():
        mask = short_report["Group"] == group

        forward_latency = short_report[mask][["Batch Size", "Forward Latency (s)"]].sort_values(by="Batch Size")
        generate_throughput = short_report[mask][["Batch Size", "Generate Throughput (tokens/s)"]].sort_values(
            by="Batch Size"
        )
        forward_memory = short_report[mask][["Batch Size", "Forward Max Memory Used (MB)"]].sort_values(
            by="Batch Size"
        )
        forward_pytorch_max_memory_allocated = short_report[mask][
            ["Batch Size", "Forward Max Memory Allocated (MB)"]
        ].sort_values(by="Batch Size")
        forward_pytorch_max_memory_reserved = short_report[mask][
            ["Batch Size", "Forward Max Memory Reserved (MB)"]
        ].sort_values(by="Batch Size")
        generate_memory = short_report[mask][["Batch Size", "Generate Max Memory Used (MB)"]].sort_values(
            by="Batch Size"
        )
        generate_pytorch_max_memory_allocated = short_report[mask][
            ["Batch Size", "Generate Max Memory Allocated (MB)"]
        ].sort_values(by="Batch Size")
        generate_pytorch_max_memory_reserved = short_report[mask][
            ["Batch Size", "Generate Max Memory Reserved (MB)"]
        ].sort_values(by="Batch Size")

        ax1.plot(
            forward_latency["Batch Size"],
            forward_latency["Forward Latency (s)"],
            label=group,
            marker="o",
        )
        ax2.plot(
            generate_throughput["Batch Size"],
            generate_throughput["Generate Throughput (tokens/s)"],
            label=group,
            marker="o",
        )
        ax3.plot(
            forward_memory["Batch Size"],
            forward_memory["Forward Max Memory Used (MB)"],
            label=group + "-used",
            marker="^",
        )
        ax3.plot(
            forward_pytorch_max_memory_reserved["Batch Size"],
            forward_pytorch_max_memory_reserved["Forward Max Memory Reserved (MB)"],
            label=group + "-reserved",
            marker=".",
        )
        ax3.plot(
            forward_pytorch_max_memory_allocated["Batch Size"],
            forward_pytorch_max_memory_allocated["Forward Max Memory Allocated (MB)"],
            label=group + "-allocated",
            marker="v",
        )
        ax4.plot(
            generate_memory["Batch Size"],
            generate_memory["Generate Max Memory Used (MB)"],
            label=group + "-used",
            marker="^",
        )
        ax4.plot(
            generate_pytorch_max_memory_reserved["Batch Size"],
            generate_pytorch_max_memory_reserved["Generate Max Memory Reserved (MB)"],
            label=group + "-reserved",
            marker=".",
        )
        ax4.plot(
            generate_pytorch_max_memory_allocated["Batch Size"],
            generate_pytorch_max_memory_allocated["Generate Max Memory Allocated (MB)"],
            label=group + "-allocated",
            marker="v",
        )

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Forward Latency (s)")
    ax1.set_title("Forward Latency per Batch Size")

    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Generate Throughput (tokens/s)")
    ax2.set_title("Generate Throughput per Batch Size")

    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("Forward Max Memory Used (MB)")
    ax3.set_title("Forward Max Memory Used per Batch Size")

    ax4.set_xlabel("Batch Size")
    ax4.set_ylabel("Generate Max Memory Used (MB)")
    ax4.set_title("Generate Max Memory Used per Batch Size")

    ax1.legend(fancybox=True, shadow=True)
    ax2.legend(fancybox=True, shadow=True)
    ax3.legend(fancybox=True, shadow=True)
    ax4.legend(fancybox=True, shadow=True)

    return fig1, fig2, fig3, fig4


def generate_report():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiments",
        "-e",
        type=Path,
        required=True,
        help="The folder containing the results of experiments.",
    )
    parser.add_argument(
        "--report-name",
        "-r",
        type=str,
        required=False,
        help="The name of the report.",
    )

    args = parser.parse_args()
    experiments_folders = args.experiments

    if args.report_name:
        report_folder = f"artifacts/{args.report_name}"
    else:
        report_folder = "artifacts"
    Path(report_folder).mkdir(parents=True, exist_ok=True)

    # gather experiments results
    inference_report = gather_inference_report(experiments_folders)
    inference_report.sort_values(by="forward.throughput(samples/s)", ascending=False, inplace=True)
    inference_report.to_csv(f"{report_folder}/full_report.csv")

    short_report = get_short_report(inference_report)
    short_report.to_csv(f"{report_folder}/short_report.csv")

    forward_throughput_plot, generate_throughput_plot, forward_memory_plot, generate_memory_plot = get_throughput_plot(
        short_report
    )
    forward_throughput_plot.savefig(f"{report_folder}/forward_latency_plot.png")
    generate_throughput_plot.savefig(f"{report_folder}/generate_throughput_plot.png")
    forward_memory_plot.savefig(f"{report_folder}/forward_memory_plot.png")
    generate_memory_plot.savefig(f"{report_folder}/generate_memory_plot.png")

    rich_table = get_rich_table(short_report)
    console = Console(record=True)
    console.print(rich_table, justify="center")
    console.save_svg(f"{report_folder}/rich_table.svg", theme=MONOKAI, title="Inference Report")


if __name__ == "__main__":
    generate_report()
