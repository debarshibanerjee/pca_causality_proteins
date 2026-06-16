#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# User-definable options.
DEFAULT_DATA_DIR = Path("analysis_outputs/ig_results/r2-70_E=1_k=50_dci=0")
DEFAULT_NUM_PCS = 5

# INPUT_CSV_NAME = "ig_curve_integrals.csv"
# Other options:
# INPUT_CSV_NAME = "ig_curve_integrals_cutoff.csv"
INPUT_CSV_NAME = "ig_curve_integrals_cutoff_significance.csv"

# FORWARD_AREA_KEY = "smoothed_pc_a_to_pc_b_area"
# REVERSE_AREA_KEY = "smoothed_pc_b_to_pc_a_area"
# Other options for ig_curve_integrals.csv:
# FORWARD_AREA_KEY = "original_pc_a_to_pc_b_area"
# REVERSE_AREA_KEY = "original_pc_b_to_pc_a_area"
# Other options for ig_curve_integrals_cutoff.csv or ig_curve_integrals_cutoff_significance.csv:
FORWARD_AREA_KEY = "original_pc_a_to_pc_b_area_to_cutoff"
REVERSE_AREA_KEY = "original_pc_b_to_pc_a_area_to_cutoff"
# FORWARD_AREA_KEY = "smoothed_pc_a_to_pc_b_area_to_cutoff"
# REVERSE_AREA_KEY = "smoothed_pc_b_to_pc_a_area_to_cutoff"

OUTPUT_NAME_WITH_VALUES = "PC_Integral_Network_Graph_color_length_dotted_smoothed_full_only_r2-70.png"
OUTPUT_NAME_WITHOUT_VALUES = "PC_Integral_Network_Graph_color_length_dotted_smoothed_full_only_no_values_r2-70.png"

SHOW_TITLE = False
# Other options:
# SHOW_TITLE = True
PLOT_TITLE = "Smoothed IG Integral Graph"
SHOW_LEGEND = True
# Other options:
# SHOW_LEGEND = False

STYLE_BLOCK_COUNT = 3
# Other options:
# STYLE_BLOCK_COUNT = 4

STYLE_BLOCKS_BY_COUNT = {
    3: (
        {
            "upper_bound": 1.0 / 3.0,
            "color": "#666666",
            "label": r"$\mathrm{IGT} < 0.33$",
            "length_mode": "fixed",
            "length_value": 0.45,
        },
        {
            "upper_bound": 2.0 / 3.0,
            "color": "#1f77b4",
            "label": r"$0.33 \leq \mathrm{IGT} < 0.67$",
            "length_mode": "fixed",
            "length_value": 0.65,
        },
        {
            "upper_bound": 1.0,
            "color": "#d62728",
            "label": r"$0.67 \leq \mathrm{IGT} \leq 1.00$",
            "length_mode": "full",
            "length_value": None,
        },
    ),
    4: (
        {
            "upper_bound": 0.25,
            "color": "#666666",
            "label": r"$\mathrm{IGT} < 0.25$",
            "length_mode": "fixed",
            "length_value": 0.35,
        },
        {
            "upper_bound": 0.50,
            "color": "#1f77b4",
            "label": r"$0.25 \leq \mathrm{IGT} < 0.50$",
            "length_mode": "fixed",
            "length_value": 0.50,
        },
        {
            "upper_bound": 0.75,
            "color": "#2ca02c",
            "label": r"$0.50 \leq \mathrm{IGT} < 0.75$",
            "length_mode": "fixed",
            "length_value": 0.70,
        },
        {
            "upper_bound": 1.0,
            "color": "#d62728",
            "label": r"$0.75 \leq \mathrm{IGT} \leq 1.00$",
            "length_mode": "full",
            "length_value": None,
        },
    ),
}

COLOR_SCHEME_SIMILARITY_THRESHOLD = 0.05
COLOR_SCHEME_SIMILARITY_POLICY = "lower"
# Other options:
# COLOR_SCHEME_SIMILARITY_POLICY = "upper"

NODE_RADIUS_VISUAL = 0.13
OFFSET_AMOUNT = 0.06
ARROW_LINE_WIDTH = 2.4
ARROW_MUTATION_SCALE = 25
VALUE_LABEL_FONT_SIZE = 9
VALUE_LABEL_OFFSET = 0.08
LEGEND_LOCATION = "center left"
LEGEND_BBOX_TO_ANCHOR = (0.88, 0.50)
FIGSIZE = (10, 8)
FIGURE_DPI = 500


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def build_smoothed_full_records(data_dir: Path, num_pcs: int) -> list[tuple[int, int, float]]:
    rows = load_csv_rows(data_dir / INPUT_CSV_NAME)
    allowed = {f"pc{i}" for i in range(1, num_pcs + 1)}

    directed_records: list[tuple[int, int, float]] = []
    for row in rows:
        pc_a = row["pc_a"]
        pc_b = row["pc_b"]
        if pc_a not in allowed or pc_b not in allowed:
            continue
        source = int(pc_a.removeprefix("pc"))
        target = int(pc_b.removeprefix("pc"))
        directed_records.append((source, target, float(row[FORWARD_AREA_KEY])))
        directed_records.append((target, source, float(row[REVERSE_AREA_KEY])))
    return directed_records


def build_positions(nodes: range) -> dict[int, np.ndarray]:
    ordered_nodes = list(nodes)
    n_nodes = len(ordered_nodes)
    if n_nodes == 0:
        return {}

    start_angle = math.pi / 2.0
    angle_step = (2.0 * math.pi) / n_nodes
    return {
        node: np.array(
            [
                math.cos(start_angle - (index * angle_step)),
                math.sin(start_angle - (index * angle_step)),
            ]
        )
        for index, node in enumerate(ordered_nodes)
    }


def normalize_records(directed_records: list[tuple[int, int, float]]) -> list[tuple[int, int, float, float]]:
    max_value = max(value for _, _, value in directed_records)
    if max_value <= 0.0:
        raise ValueError("Cannot normalize a dataset whose maximum integral is <= 0.")
    return [(source, target, raw_value, raw_value / max_value) for source, target, raw_value in directed_records]


def get_style_blocks() -> tuple[dict[str, object], ...]:
    try:
        return STYLE_BLOCKS_BY_COUNT[STYLE_BLOCK_COUNT]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported STYLE_BLOCK_COUNT={STYLE_BLOCK_COUNT}. " f"Choose one of: {sorted(STYLE_BLOCKS_BY_COUNT)}."
        ) from exc


def get_style_boundaries(style_blocks: tuple[dict[str, object], ...]) -> tuple[float, ...]:
    return tuple(float(block["upper_bound"]) for block in style_blocks[:-1])


def class_index_for_value(value: float, style_blocks: tuple[dict[str, object], ...]) -> int:
    for class_index, block in enumerate(style_blocks[:-1]):
        if value < float(block["upper_bound"]):
            return class_index
    return len(style_blocks) - 1


def resolve_color_classes(
    normalized_records: list[tuple[int, int, float, float]],
) -> list[int]:
    if COLOR_SCHEME_SIMILARITY_POLICY not in {"lower", "upper"}:
        raise ValueError("COLOR_SCHEME_SIMILARITY_POLICY must be 'lower' or 'upper'.")

    style_blocks = get_style_blocks()
    style_boundaries = get_style_boundaries(style_blocks)
    normalized_values = [normalized_value for _, _, _, normalized_value in normalized_records]
    base_classes = [class_index_for_value(value, style_blocks) for value in normalized_values]
    if COLOR_SCHEME_SIMILARITY_THRESHOLD <= 0.0 or len(normalized_records) < 2:
        return base_classes

    resolved_classes = list(base_classes)

    for pair_start in range(0, len(normalized_records), 2):
        pair_end = pair_start + 1
        if pair_end >= len(normalized_records):
            break

        source_a, target_a, _, value_a = normalized_records[pair_start]
        source_b, target_b, _, value_b = normalized_records[pair_end]
        if source_a != target_b or target_a != source_b:
            continue

        class_a = base_classes[pair_start]
        class_b = base_classes[pair_end]
        if class_a == class_b:
            continue

        lower_value = min(value_a, value_b)
        upper_value = max(value_a, value_b)

        for boundary_index, boundary in enumerate(style_boundaries):
            if not (lower_value < boundary <= upper_value):
                continue
            if abs(value_a - boundary) > COLOR_SCHEME_SIMILARITY_THRESHOLD:
                continue
            if abs(value_b - boundary) > COLOR_SCHEME_SIMILARITY_THRESHOLD:
                continue

            resolved_class = boundary_index if COLOR_SCHEME_SIMILARITY_POLICY == "lower" else boundary_index + 1
            resolved_classes[pair_start] = resolved_class
            resolved_classes[pair_end] = resolved_class
            break

    return resolved_classes


def add_edge_value_label(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    perp_vec: np.ndarray,
    normalized_value: float,
) -> None:
    midpoint = 0.5 * (start + end)
    text_pos = midpoint + (VALUE_LABEL_OFFSET * perp_vec)
    ax.text(
        text_pos[0],
        text_pos[1],
        f"{normalized_value:.2f}",
        fontsize=VALUE_LABEL_FONT_SIZE,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.2},
        zorder=4,
    )


def output_path_for(data_dir: Path, show_values: bool) -> Path:
    return data_dir / (OUTPUT_NAME_WITH_VALUES if show_values else OUTPUT_NAME_WITHOUT_VALUES)


def print_normalized_integrals(normalized_records: list[tuple[int, int, float, float]]) -> None:
    print("Normalized directed integrals:")
    for pair_start in range(0, len(normalized_records), 2):
        pair = normalized_records[pair_start : pair_start + 2]
        if len(pair) < 2:
            source, target, _, normalized_value = pair[0]
            print(f"PC{source} -> PC{target}: {normalized_value:.2f}")
            continue

        source_a, target_a, _, normalized_a = pair[0]
        source_b, target_b, _, normalized_b = pair[1]
        if source_a == target_b and target_a == source_b:
            print(
                f"PC{source_a} <-> PC{target_a}: PC{source_a}->PC{target_a}={normalized_a:.2f}, PC{target_a}->PC{source_a}={normalized_b:.2f}"
            )
        else:
            print(f"PC{source_a} -> PC{target_a}: {normalized_a:.2f}")
            print(f"PC{source_b} -> PC{target_b}: {normalized_b:.2f}")


def draw_graph(
    data_dir: Path,
    normalized_records: list[tuple[int, int, float, float]],
    num_pcs: int,
    show_values: bool,
) -> Path:
    style_blocks = get_style_blocks()
    style_classes = resolve_color_classes(normalized_records)
    nodes = range(1, num_pcs + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIGURE_DPI)
    pos = build_positions(nodes)

    for node, center in pos.items():
        circle = plt.Circle(
            center,
            radius=NODE_RADIUS_VISUAL,
            facecolor="whitesmoke",
            edgecolor="black",
            linewidth=2,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            center[0],
            center[1],
            rf"$\mathrm{{PC}}_{node}$",
            fontsize=12,
            fontweight="bold",
            fontfamily="sans-serif",
            ha="center",
            va="center",
            zorder=4,
        )

    for edge_index, (source, target, _, normalized_value) in enumerate(normalized_records):
        if normalized_value <= 0.0:
            continue

        point_a = np.array(pos[source])
        point_b = np.array(pos[target])
        vec = point_b - point_a
        dist = np.linalg.norm(vec)
        if dist == 0.0:
            continue
        unit_vec = vec / dist
        perp_vec = np.array([unit_vec[1], -unit_vec[0]])

        start_shifted = point_a + (perp_vec * OFFSET_AMOUNT)
        end_shifted = point_b + (perp_vec * OFFSET_AMOUNT)

        draw_start = start_shifted + (unit_vec * NODE_RADIUS_VISUAL)
        max_draw_end = end_shifted - (unit_vec * NODE_RADIUS_VISUAL)
        full_drawable_vec = max_draw_end - draw_start
        full_drawable_len = np.linalg.norm(full_drawable_vec)
        if full_drawable_len == 0.0:
            continue

        style_class = style_classes[edge_index]
        style_block = style_blocks[style_class]
        edge_color = str(style_block["color"])
        if style_block["length_mode"] == "fixed":
            actual_len = min(float(style_block["length_value"]), full_drawable_len)
            is_full_length = False
        else:
            actual_len = full_drawable_len
            is_full_length = True

        draw_end = draw_start + (unit_vec * actual_len)

        if not is_full_length:
            ax.plot(
                [draw_end[0], max_draw_end[0]],
                [draw_end[1], max_draw_end[1]],
                linestyle=":",
                color=edge_color,
                linewidth=max(1.0, ARROW_LINE_WIDTH - 0.4),
                alpha=0.9,
                zorder=1,
            )

        arrow = patches.FancyArrowPatch(
            posA=draw_start,
            posB=draw_end,
            connectionstyle="arc3,rad=0",
            color=edge_color,
            arrowstyle="-|>",
            mutation_scale=ARROW_MUTATION_SCALE,
            linewidth=ARROW_LINE_WIDTH,
            alpha=0.95,
            zorder=2,
        )
        ax.add_patch(arrow)

        if show_values:
            add_edge_value_label(ax, draw_start, draw_end, perp_vec, normalized_value)

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    if SHOW_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")

    if SHOW_LEGEND:
        legend_elements = [
            Line2D([0], [0], color=str(block["color"]), lw=2.5, label=str(block["label"])) for block in style_blocks
        ]
        ax.legend(
            handles=legend_elements,
            loc=LEGEND_LOCATION,
            bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
            frameon=True,
        )

    output_path = output_path_for(data_dir, show_values)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create color-and-length dotted PC interaction network graphs from the " "smoothed full-integral IG dataset only."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing ig_curve_integrals.csv.",
    )
    parser.add_argument(
        "--num-pcs",
        type=int,
        default=DEFAULT_NUM_PCS,
        help="Number of PCs to include, starting from PC1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    directed_records = build_smoothed_full_records(data_dir, args.num_pcs)
    if not directed_records:
        raise SystemExit("No smoothed full-integral records were found for the requested PC range.")

    normalized_records = normalize_records(directed_records)
    print_normalized_integrals(normalized_records)

    outputs = [
        draw_graph(
            data_dir=data_dir,
            normalized_records=normalized_records,
            num_pcs=args.num_pcs,
            show_values=True,
        ),
        draw_graph(
            data_dir=data_dir,
            normalized_records=normalized_records,
            num_pcs=args.num_pcs,
            show_values=False,
        ),
    ]

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
