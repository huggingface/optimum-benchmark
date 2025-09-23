from typing import List, Tuple

import matplotlib.pyplot as plt

from .benchmark.report import BenchmarkReport
from .trackers.latency import Latency, Throughput


def _get_color_palette(n: int) -> List[str]:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % cmap.N) for i in range(n)]


def _get_latency_unit_and_values(
    all_values: List[List[float]], means: List[float]
) -> Tuple[str, List[List[float]], List[float]]:
    """Determine appropriate unit (ms or s) and convert values accordingly.

    Returns:
        unit_label: "ms" or "s"
        converted_values: values converted to the chosen unit
        converted_means: means converted to the chosen unit
    """
    # Check if all values (including means) are < 1.0 seconds
    all_data = []
    for vals in all_values:
        all_data.extend(vals)
    all_data.extend(means)

    if all(val < 1.0 for val in all_data):
        converted_values = [[val * 1000 for val in vals] for vals in all_values]
        converted_means = [mean * 1000 for mean in means]
        return "ms", converted_values, converted_means
    else:
        return "s", all_values, means


def _setup_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str, rotate_xticks: int) -> None:
    """Apply common axis styling: title, labels, and rotated x-tick labels."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for label in ax.get_xticklabels():
        label.set_rotation(rotate_xticks)
        label.set_ha("right")


def _create_boxplot_with_overlay(
    ax: plt.Axes, values: List[List[float]], means: List[float], names: List[str], colors: List[str]
) -> None:
    """Create boxplot with semi-transparent bar overlay and mean annotations."""
    # Create boxplot
    bp = ax.boxplot(values, patch_artist=True, labels=names, showfliers=False)

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Overlay bar plot with means (semi-transparent)
    x_positions = range(1, len(names) + 1)  # boxplot positions are 1-indexed
    ax.bar(x_positions, means, alpha=0.3, color=colors, width=0.6)

    # Annotate mean values on top of boxplots
    for i, mean_val in enumerate(means):
        ax.annotate(
            f"{mean_val:.3f}",
            xy=(i + 1, mean_val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _create_bar_plot(
    ax: plt.Axes, names: List[str], values: List[float], colors: List[str], annotate: bool = True
) -> None:
    """Create bar plot with optional value annotations."""
    bars = ax.bar(names, values, color=colors)
    if annotate:
        ax.bar_label(bars, padding=3, fmt="%.2f")


def _extract_latency_data(
    reports: dict[str, BenchmarkReport], attr_path: str
) -> Tuple[List[str], List[float], List[List[float]]]:
    """Extract latency data from reports for a given attribute path (e.g., 'prefill', 'per_token', 'forward')."""
    names = list(reports.keys())
    means = []
    values = []

    for name in names:
        latency_obj = getattr(reports[name], attr_path).latency
        means.append(latency_obj.mean)
        values.append(latency_obj.values)

    return names, means, values


def _extract_throughput_data(reports: dict[str, BenchmarkReport], attr_path: str) -> Tuple[List[str], List[float]]:
    """Extract throughput data from reports for a given attribute path (e.g., 'decode', 'forward')."""
    names = list(reports.keys())
    values = []

    for name in names:
        throughput_obj = getattr(reports[name], attr_path).throughput
        values.append(throughput_obj.value)

    return names, values


def plot_prefill_latencies(
    reports: dict[str, BenchmarkReport], figsize: Tuple[int, int] = (10, 6)
) -> tuple[plt.Figure, plt.Axes]:
    assert len(reports) > 1, "Need at least two reports to plot comparison."
    assert all(
        hasattr(report, "prefill")
        and hasattr(report.prefill, "latency")
        and isinstance(report.prefill.latency, Latency)
        for report in reports.values()
    ), "All reports must have prefill latency measurements."

    # Extract data
    names, means, values = _extract_latency_data(reports, "prefill")

    # Determine appropriate unit and convert values
    unit, plot_values, plot_means = _get_latency_unit_and_values(values, means)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    colors = _get_color_palette(len(names))

    # Create boxplot with overlay
    _create_boxplot_with_overlay(ax, plot_values, plot_means, names, colors)

    # Setup axes
    _setup_axes(ax, "Prefill Latencies", "Configurations", f"Latency ({unit})", rotate_xticks=30)

    return fig, ax


def plot_per_token_latencies(reports, figsize: Tuple[int, int] = (10, 6)):
    assert len(reports) > 1, "Need at least two reports to plot comparison."
    assert all(
        hasattr(report, "per_token")
        and hasattr(report.per_token, "latency")
        and isinstance(report.per_token.latency, Latency)
        for report in reports.values()
    ), "All reports must have per-token latency measurements."

    # Extract data
    names, means, values = _extract_latency_data(reports, "per_token")

    # Determine appropriate unit and convert values
    unit, plot_values, plot_means = _get_latency_unit_and_values(values, means)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    colors = _get_color_palette(len(names))

    # Create boxplot with overlay
    _create_boxplot_with_overlay(ax, plot_values, plot_means, names, colors)

    # Setup axes
    _setup_axes(ax, "Per-Token Latencies", "Configurations", f"Latency ({unit})", rotate_xticks=30)

    return fig, ax


def plot_decode_throughputs(reports, figsize: Tuple[int, int] = (10, 6), annotate: bool = True):
    assert len(reports) > 1, "Need at least two reports to plot comparison."
    assert all(
        hasattr(report, "decode")
        and hasattr(report.decode, "throughput")
        and isinstance(report.decode.throughput, Throughput)
        for report in reports.values()
    ), "All reports must have decode throughput measurements."

    # Extract data
    names, values = _extract_throughput_data(reports, "decode")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    colors = _get_color_palette(len(names))

    # Create bar plot
    _create_bar_plot(ax, names, values, colors, annotate)

    # Setup axes
    _setup_axes(ax, "Decoding Throughput", "Configurations", "Throughput (tokens/s)", rotate_xticks=30)

    return fig, ax


def plot_forward_latencies(reports, figsize: Tuple[int, int] = (10, 6)):
    assert len(reports) > 1, "Need at least two reports to plot comparison."
    assert all(
        hasattr(report, "forward")
        and hasattr(report.forward, "latency")
        and isinstance(report.forward.latency, Latency)
        for report in reports.values()
    ), "All reports must have forward latency measurements."

    # Extract data
    names, means, values = _extract_latency_data(reports, "forward")

    # Determine appropriate unit and convert values
    unit, plot_values, plot_means = _get_latency_unit_and_values(values, means)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    colors = _get_color_palette(len(names))

    # Create boxplot with overlay
    _create_boxplot_with_overlay(ax, plot_values, plot_means, names, colors)

    # Setup axes
    _setup_axes(ax, "Forward Pass Latencies", "Configurations", f"Latency ({unit})", rotate_xticks=30)

    return fig, ax


def plot_forward_throughputs(reports, figsize: Tuple[int, int] = (10, 6), annotate: bool = True):
    assert len(reports) > 1, "Need at least two reports to plot comparison."
    assert all(
        hasattr(report, "forward")
        and hasattr(report.forward, "throughput")
        and isinstance(report.forward.throughput, Throughput)
        for report in reports.values()
    ), "All reports must have forward throughput measurements."

    # Extract data
    names, values = _extract_throughput_data(reports, "forward")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    colors = _get_color_palette(len(names))

    # Create bar plot
    _create_bar_plot(ax, names, values, colors, annotate)

    # Setup axes
    _setup_axes(ax, "Forward Pass Throughput", "Configurations", "Throughput (samples/s)", rotate_xticks=30)

    return fig, ax
