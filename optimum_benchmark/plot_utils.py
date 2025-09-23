import matplotlib.pyplot as plt

from .benchmark.report import BenchmarkReport
from .trackers.latency import Latency, Throughput


def plot_prefill_latencies(reports: dict[str, BenchmarkReport]) -> tuple[plt.Figure, plt.Axes]:
    assert len(reports) > 1, "Need at least two reports to plot comparison."
    assert all(
        hasattr(report, "prefill")
        and hasattr(report.prefill, "latency")
        and isinstance(report.prefill.latency, Latency)
        for report in reports.values()
    ), "All reports must have prefill latency measurements."

    fig, ax = plt.subplots()
    ax.boxplot(
        [reports[config_name].prefill.latency.values for config_name in reports.keys()], tick_labels=reports.keys()
    )
    ax.set_title("Prefill Latencies")
    ax.set_xlabel("Configurations")
    ax.set_ylabel("Latency (s)")

    return fig, ax


def plot_per_token_latencies(reports):
    assert len(reports) > 1, "Need at least two reports to plot comparison."
    assert all(
        hasattr(report, "per_token")
        and hasattr(report.per_token, "latency")
        and isinstance(report.per_token.latency, Latency)
        for report in reports.values()
    ), "All reports must have per-token latency measurements."

    fig, ax = plt.subplots()
    ax.boxplot(
        [reports[config_name].per_token.latency.values for config_name in reports.keys()], tick_labels=reports.keys()
    )
    ax.set_title("Per-Token Latencies")
    ax.set_xlabel("Configurations")
    ax.set_ylabel("Latency (s)")

    return fig, ax


def plot_decode_throughputs(reports):
    assert len(reports) > 1, "Need at least two reports to plot comparison."
    assert all(
        hasattr(report, "decode")
        and hasattr(report.decode, "throughput")
        and isinstance(report.decode.throughput, Throughput)
        for report in reports.values()
    ), "All reports must have decode throughput measurements."

    fig, ax = plt.subplots()
    ax.bar(list(reports.keys()), [reports[config_name].decode.throughput.value for config_name in reports.keys()])
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Decoding Throughput")
    ax.set_xlabel("Configurations")

    return fig, ax


def plot_forward_latencies(reports):
    assert len(reports) > 1, "Need at least two reports to plot comparison."
    assert all(
        hasattr(report, "forward")
        and hasattr(report.forward, "latency")
        and isinstance(report.forward.latency, Latency)
        for report in reports.values()
    ), "All reports must have forward latency measurements."

    fig, ax = plt.subplots()
    ax.boxplot(
        [reports[config_name].forward.latency.values for config_name in reports.keys()], tick_labels=reports.keys()
    )
    ax.set_title("Forward Pass Latencies")
    ax.set_ylabel("Latency (s)")
    ax.set_xlabel("Configurations")

    return fig, ax


def plot_forward_throughputs(reports):
    assert len(reports) > 1, "Need at least two reports to plot comparison."
    assert all(
        hasattr(report, "forward")
        and hasattr(report.forward, "throughput")
        and isinstance(report.forward.throughput, Throughput)
        for report in reports.values()
    ), "All reports must have forward throughput measurements."

    fig, ax = plt.subplots()
    ax.bar(list(reports.keys()), [reports[config_name].forward.throughput.value for config_name in reports.keys()])
    ax.set_title("Forward Pass Throughput")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_xlabel("Configurations")

    return fig, ax
