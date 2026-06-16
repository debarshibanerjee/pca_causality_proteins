#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee", "no-latex", "std-colors"])
# plt.style.use("default")
plt.rcParams["font.size"] = 32
# colors = ["#1982C4", "#111111", "#FF595E", "#20B806"]

# User-definable options.
# DEFAULT_INPUT_DIR = Path("analysis_outputs/ig_results/200to490us_sklearn_standard_top5/E=1_k=50_dci=0")
DEFAULT_INPUT_DIR = Path("analysis_outputs/ig_results/830usto1060us_sklearn_standard_top5/E=1_k=50_dci=0")
PC_PAIRS_TO_PLOT = ((1, 2), (1, 3), (1, 4), (1, 5))
GRID_SHAPE = (2, 2)
OUTPUT_FILENAME = "IG_selected_pairs_smoothed_grid.png"

FORWARD_COLOR = "#D81B60"
REVERSE_COLOR = "#20B806"

FIGSIZE = (20, 12)
FIGURE_DPI = 500
LINE_WIDTH = 3.6
SHOW_SIGNIFICANCE_MARKERS = False
# Other options:
# SHOW_SIGNIFICANCE_MARKERS = True
MARKER_SIZE = 6
SHADE_ALPHA = 0.2
X_AXIS_MAX_NS = 1000.0
X_TICKS_NS = np.arange(0.0, 1001.0, 200.0)
Y_AXIS_LABEL = "Imbalance Gain [%]"
BOTTOM_X_AXIS_LABEL = r"Time Lag $\tau$ (ns)"
ZERO_LINE_COLOR = "0.75"
ZERO_LINE_STYLE = "--"
ZERO_LINE_WIDTH = 1.0
ZERO_LINE_ALPHA = 0.8
Y_TICK_NBINS = 6
Y_TICK_MIN = 5
LEGEND_FONT_SIZE = 28
TICK_FONT_SIZE = 30
AXIS_LABEL_FONT_SIZE = 34
SHARED_X_LABEL_Y = -0.02
SHARED_Y_LABEL_X = -0.02

from ig_smoothing_common import (
    DEFAULT_INTERPOLATOR,
    DEFAULT_INTERP_POINTS,
    DEFAULT_PASSES,
    DEFAULT_POLYORDER,
    DEFAULT_WINDOW_LENGTH,
    find_closest_indices,
    load_ig_table,
    parse_pair_metadata,
    smooth_and_interpolate,
)


def format_pc_label(raw_label: str) -> str:
    match = re.fullmatch(r"pc(\d+)", raw_label)
    if match is None:
        return rf"\mathrm{{{raw_label.upper()}}}"
    return rf"\mathrm{{PC}}_{match.group(1)}"


def arrow_label(source: str, target: str) -> str:
    return rf"${source} \rightarrow {target}$"


def pair_title(pc_a: str, pc_b: str) -> str:
    return rf"${format_pc_label(pc_a)} \mathrm{{vs.}} {format_pc_label(pc_b)}$"


def build_shared_label_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_shared_axis_labels{output_path.suffix}")


def build_pair_input_path(input_dir: Path, pair: tuple[int, int]) -> Path:
    pc_a, pc_b = pair
    pattern = f"IG-pc{pc_a}_vs_pc{pc_b}.txt"
    matches = sorted(input_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No IG file matched {pattern!r} in {input_dir}.")
    if len(matches) > 1:
        raise FileExistsError(f"Multiple IG files matched {pattern!r} in {input_dir}: {[path.name for path in matches]}")
    return matches[0]


def compute_smoothed_series(
    input_path: Path,
    num_interp_points: int,
    smooth_window_length: int,
    smooth_poly_deg: int,
    num_passes: int,
    interpolating_func: str,
) -> dict[str, object]:
    data = load_ig_table(input_path)
    metadata = parse_pair_metadata(input_path)

    taus = data[:, 0]
    mask_a_to_b = data[:, 3].astype(bool)
    mask_b_to_a = data[:, 4].astype(bool)

    smooth_a_to_b = smooth_and_interpolate(
        data[:, [0, 1]],
        num_interp_points=num_interp_points,
        smooth_window_length=smooth_window_length,
        smooth_poly_deg=smooth_poly_deg,
        num_passes=num_passes,
        interpolating_func=interpolating_func,
    )
    smooth_b_to_a = smooth_and_interpolate(
        data[:, [0, 2]],
        num_interp_points=num_interp_points,
        smooth_window_length=smooth_window_length,
        smooth_poly_deg=smooth_poly_deg,
        num_passes=num_passes,
        interpolating_func=interpolating_func,
    )
    smooth_err_a_to_b = smooth_and_interpolate(
        data[:, [0, 5]],
        num_interp_points=num_interp_points,
        smooth_window_length=smooth_window_length,
        smooth_poly_deg=smooth_poly_deg,
        num_passes=num_passes,
        interpolating_func=interpolating_func,
    )
    smooth_err_b_to_a = smooth_and_interpolate(
        data[:, [0, 6]],
        num_interp_points=num_interp_points,
        smooth_window_length=smooth_window_length,
        smooth_poly_deg=smooth_poly_deg,
        num_passes=num_passes,
        interpolating_func=interpolating_func,
    )

    significant_a_to_b = taus[mask_a_to_b]
    significant_b_to_a = taus[mask_b_to_a]
    closest_a_to_b = find_closest_indices(smooth_a_to_b[:, 0], significant_a_to_b)
    closest_b_to_a = find_closest_indices(smooth_b_to_a[:, 0], significant_b_to_a)

    return {
        "metadata": metadata,
        "taus": taus,
        "smooth_a_to_b": smooth_a_to_b,
        "smooth_b_to_a": smooth_b_to_a,
        "smooth_err_a_to_b": smooth_err_a_to_b,
        "smooth_err_b_to_a": smooth_err_b_to_a,
        "significant_a_to_b": significant_a_to_b,
        "significant_b_to_a": significant_b_to_a,
        "closest_a_to_b": closest_a_to_b,
        "closest_b_to_a": closest_b_to_a,
    }


def plot_pair_on_axis(
    ax: plt.Axes,
    series: dict[str, object],
    show_xlabel: bool,
    show_axis_labels: bool,
) -> None:
    from matplotlib.ticker import MaxNLocator

    metadata = series["metadata"]
    pc_a_label = format_pc_label(metadata["pc_a"])
    pc_b_label = format_pc_label(metadata["pc_b"])

    smooth_a_to_b = series["smooth_a_to_b"]
    smooth_b_to_a = series["smooth_b_to_a"]
    smooth_err_a_to_b = series["smooth_err_a_to_b"]
    smooth_err_b_to_a = series["smooth_err_b_to_a"]
    significant_a_to_b = series["significant_a_to_b"]
    significant_b_to_a = series["significant_b_to_a"]
    closest_a_to_b = series["closest_a_to_b"]
    closest_b_to_a = series["closest_b_to_a"]
    taus = series["taus"]

    ax.plot(
        smooth_a_to_b[:, 0],
        smooth_a_to_b[:, 1],
        "-",
        linewidth=LINE_WIDTH,
        color=FORWARD_COLOR,
        label=arrow_label(pc_a_label, pc_b_label),
    )
    ax.plot(
        smooth_b_to_a[:, 0],
        smooth_b_to_a[:, 1],
        "-",
        linewidth=LINE_WIDTH,
        color=REVERSE_COLOR,
        label=arrow_label(pc_b_label, pc_a_label),
    )

    if SHOW_SIGNIFICANCE_MARKERS and significant_a_to_b.size:
        ax.plot(
            significant_a_to_b,
            smooth_a_to_b[closest_a_to_b, 1],
            "o",
            color=FORWARD_COLOR,
            markersize=MARKER_SIZE,
        )
    if SHOW_SIGNIFICANCE_MARKERS and significant_b_to_a.size:
        ax.plot(
            significant_b_to_a,
            smooth_b_to_a[closest_b_to_a, 1],
            "o",
            color=REVERSE_COLOR,
            markersize=MARKER_SIZE,
        )

    ax.fill_between(
        smooth_a_to_b[:, 0],
        smooth_a_to_b[:, 1] - smooth_err_a_to_b[:, 1],
        smooth_a_to_b[:, 1] + smooth_err_a_to_b[:, 1],
        alpha=SHADE_ALPHA,
        color=FORWARD_COLOR,
    )
    ax.fill_between(
        smooth_b_to_a[:, 0],
        smooth_b_to_a[:, 1] - smooth_err_b_to_a[:, 1],
        smooth_b_to_a[:, 1] + smooth_err_b_to_a[:, 1],
        alpha=SHADE_ALPHA,
        color=REVERSE_COLOR,
    )
    ax.axhline(
        0.0,
        color=ZERO_LINE_COLOR,
        linestyle=ZERO_LINE_STYLE,
        linewidth=ZERO_LINE_WIDTH,
        alpha=ZERO_LINE_ALPHA,
        zorder=0,
    )

    y_values = np.concatenate(
        [
            smooth_a_to_b[:, 1] - smooth_err_a_to_b[:, 1],
            smooth_a_to_b[:, 1] + smooth_err_a_to_b[:, 1],
            smooth_b_to_a[:, 1] - smooth_err_b_to_a[:, 1],
            smooth_b_to_a[:, 1] + smooth_err_b_to_a[:, 1],
        ]
    )
    y_padding = max(0.05, 0.08 * (y_values.max() - y_values.min() or 1.0))

    ax.set_xlim(float(taus.min()), X_AXIS_MAX_NS)
    ax.set_xticks(X_TICKS_NS)
    ax.set_ylim(y_values.min() - y_padding, y_values.max() + y_padding)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=Y_TICK_NBINS, min_n_ticks=Y_TICK_MIN))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    if show_axis_labels:
        ax.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    else:
        ax.set_ylabel("")
    ax.legend(frameon=True, fontsize=LEGEND_FONT_SIZE)

    if show_xlabel:
        if show_axis_labels:
            ax.set_xlabel(BOTTOM_X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
        else:
            ax.set_xlabel("")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)


def plot_smoothed_ig_grid(
    input_dir: Path,
    output_filename: str,
    num_interp_points: int,
    smooth_window_length: int,
    smooth_poly_deg: int,
    num_passes: int,
    interpolating_func: str,
) -> tuple[Path, Path]:
    if len(PC_PAIRS_TO_PLOT) != GRID_SHAPE[0] * GRID_SHAPE[1]:
        raise ValueError(
            f"PC_PAIRS_TO_PLOT must contain exactly {GRID_SHAPE[0] * GRID_SHAPE[1]} pairs for a {GRID_SHAPE[0]}x{GRID_SHAPE[1]} grid."
        )

    series_by_pair = [
        compute_smoothed_series(
            input_path=build_pair_input_path(input_dir, pair),
            num_interp_points=num_interp_points,
            smooth_window_length=smooth_window_length,
            smooth_poly_deg=smooth_poly_deg,
            num_passes=num_passes,
            interpolating_func=interpolating_func,
        )
        for pair in PC_PAIRS_TO_PLOT
    ]

    fig, axes = plt.subplots(
        GRID_SHAPE[0],
        GRID_SHAPE[1],
        figsize=FIGSIZE,
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )
    flat_axes = list(np.ravel(axes))

    for index, (ax, series) in enumerate(zip(flat_axes, series_by_pair, strict=True)):
        row_index = index // GRID_SHAPE[1]
        show_xlabel = row_index == GRID_SHAPE[0] - 1
        plot_pair_on_axis(ax=ax, series=series, show_xlabel=show_xlabel, show_axis_labels=True)

    output_path = input_dir / output_filename
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    shared_label_fig, shared_label_axes = plt.subplots(
        GRID_SHAPE[0],
        GRID_SHAPE[1],
        figsize=FIGSIZE,
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )
    flat_shared_axes = list(np.ravel(shared_label_axes))

    for index, (ax, series) in enumerate(zip(flat_shared_axes, series_by_pair, strict=True)):
        row_index = index // GRID_SHAPE[1]
        show_xlabel = row_index == GRID_SHAPE[0] - 1
        plot_pair_on_axis(ax=ax, series=series, show_xlabel=show_xlabel, show_axis_labels=False)

    shared_label_fig.text(
        0.5,
        SHARED_X_LABEL_Y,
        BOTTOM_X_AXIS_LABEL,
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    shared_label_fig.text(
        SHARED_Y_LABEL_X,
        0.5,
        Y_AXIS_LABEL,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )

    shared_label_output_path = build_shared_label_output_path(output_path)
    shared_label_fig.savefig(shared_label_output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(shared_label_fig)
    return output_path, shared_label_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 2x2 grid of smoothed IG plots for four selected PC pairs.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing IG-*.txt files.",
    )
    parser.add_argument(
        "--output-filename",
        default=OUTPUT_FILENAME,
        help="Filename for the 2x2 grid PNG written inside input_dir.",
    )
    parser.add_argument(
        "--interp-points",
        type=int,
        default=DEFAULT_INTERP_POINTS,
        help="Number of interpolation points for the smoothed curves.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=DEFAULT_WINDOW_LENGTH,
        help="Savitzky-Golay window length before any automatic adjustment.",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=DEFAULT_POLYORDER,
        help="Savitzky-Golay polynomial degree.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=DEFAULT_PASSES,
        help="Number of repeated smoothing passes.",
    )
    parser.add_argument(
        "--interpolator",
        choices=["linear", "quadratic", "cubic"],
        default=DEFAULT_INTERPOLATOR,
        help="Interpolation kind used after smoothing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_paths = plot_smoothed_ig_grid(
        input_dir=input_dir,
        output_filename=args.output_filename,
        num_interp_points=args.interp_points,
        smooth_window_length=args.window_length,
        smooth_poly_deg=args.polyorder,
        num_passes=args.passes,
        interpolating_func=args.interpolator,
    )
    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
