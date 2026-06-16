#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ig_smoothing_common import (
    DEFAULT_INTERPOLATOR,
    DEFAULT_INTERP_POINTS,
    DEFAULT_PASSES,
    DEFAULT_POLYORDER,
    DEFAULT_WINDOW_LENGTH,
    first_nonpositive_lower_bound_time,
    integrate_positive_curve_with_method,
    integrate_positive_curve_up_to,
    load_ig_table,
    parse_pair_metadata,
    smooth_and_interpolate,
)


def integrate_pair(
    input_path: Path,
    num_interp_points: int,
    smooth_window_length: int,
    smooth_poly_deg: int,
    num_passes: int,
    interpolating_func: str,
    integration_method: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = load_ig_table(input_path)
    metadata = parse_pair_metadata(input_path)
    taus = data[:, 0]

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

    original_cutoff_a_to_b = first_nonpositive_lower_bound_time(taus, data[:, 1], data[:, 5])
    original_cutoff_b_to_a = first_nonpositive_lower_bound_time(taus, data[:, 2], data[:, 6])
    smoothed_cutoff_a_to_b = first_nonpositive_lower_bound_time(smooth_a_to_b[:, 0], smooth_a_to_b[:, 1], smooth_err_a_to_b[:, 1])
    smoothed_cutoff_b_to_a = first_nonpositive_lower_bound_time(smooth_b_to_a[:, 0], smooth_b_to_a[:, 1], smooth_err_b_to_a[:, 1])
    original_cutoff_found_a_to_b = original_cutoff_a_to_b is not None
    original_cutoff_found_b_to_a = original_cutoff_b_to_a is not None
    smoothed_cutoff_found_a_to_b = smoothed_cutoff_a_to_b is not None
    smoothed_cutoff_found_b_to_a = smoothed_cutoff_b_to_a is not None
    original_effective_cutoff_a_to_b = original_cutoff_a_to_b if original_cutoff_found_a_to_b else float(taus[-1])
    original_effective_cutoff_b_to_a = original_cutoff_b_to_a if original_cutoff_found_b_to_a else float(taus[-1])
    smoothed_effective_cutoff_a_to_b = smoothed_cutoff_a_to_b if smoothed_cutoff_found_a_to_b else float(smooth_a_to_b[-1, 0])
    smoothed_effective_cutoff_b_to_a = smoothed_cutoff_b_to_a if smoothed_cutoff_found_b_to_a else float(smooth_b_to_a[-1, 0])

    original_low_a_to_b = data[:, 1] - data[:, 5]
    original_low_b_to_a = data[:, 2] - data[:, 6]
    original_high_a_to_b = data[:, 1] + data[:, 5]
    original_high_b_to_a = data[:, 2] + data[:, 6]
    smoothed_low_a_to_b = smooth_a_to_b[:, 1] - smooth_err_a_to_b[:, 1]
    smoothed_low_b_to_a = smooth_b_to_a[:, 1] - smooth_err_b_to_a[:, 1]
    smoothed_high_a_to_b = smooth_a_to_b[:, 1] + smooth_err_a_to_b[:, 1]
    smoothed_high_b_to_a = smooth_b_to_a[:, 1] + smooth_err_b_to_a[:, 1]

    return (
        {
            "pair": f"{metadata['pc_a']}-{metadata['pc_b']}",
            "pc_a": metadata["pc_a"],
            "pc_b": metadata["pc_b"],
            "tau_start_ns": float(taus.min()),
            "tau_stop_ns": float(taus.max()),
            "original_pc_a_to_pc_b_area": integrate_positive_curve_with_method(taus, data[:, 1], method=integration_method),
            "original_pc_b_to_pc_a_area": integrate_positive_curve_with_method(taus, data[:, 2], method=integration_method),
            "smoothed_pc_a_to_pc_b_area": integrate_positive_curve_with_method(
                smooth_a_to_b[:, 0], smooth_a_to_b[:, 1], method=integration_method
            ),
            "smoothed_pc_b_to_pc_a_area": integrate_positive_curve_with_method(
                smooth_b_to_a[:, 0], smooth_b_to_a[:, 1], method=integration_method
            ),
            "original_pc_a_to_pc_b_positive_points": int((data[:, 1] > 0.0).sum()),
            "original_pc_b_to_pc_a_positive_points": int((data[:, 2] > 0.0).sum()),
            "smoothed_pc_a_to_pc_b_positive_points": int((smooth_a_to_b[:, 1] > 0.0).sum()),
            "smoothed_pc_b_to_pc_a_positive_points": int((smooth_b_to_a[:, 1] > 0.0).sum()),
            "original_n_points": int(len(taus)),
            "smoothed_n_points": int(len(smooth_a_to_b)),
        },
        {
            "pair": f"{metadata['pc_a']}-{metadata['pc_b']}",
            "pc_a": metadata["pc_a"],
            "pc_b": metadata["pc_b"],
            "tau_start_ns": float(taus.min()),
            "tau_stop_ns": float(taus.max()),
            "original_pc_a_to_pc_b_cutoff_found": original_cutoff_found_a_to_b,
            "original_pc_b_to_pc_a_cutoff_found": original_cutoff_found_b_to_a,
            "original_pc_a_to_pc_b_cutoff_ns": original_effective_cutoff_a_to_b,
            "original_pc_b_to_pc_a_cutoff_ns": original_effective_cutoff_b_to_a,
            "original_pc_a_to_pc_b_area_to_cutoff": integrate_positive_curve_up_to(
                taus,
                data[:, 1],
                original_effective_cutoff_a_to_b,
                method=integration_method,
            ),
            "original_pc_b_to_pc_a_area_to_cutoff": integrate_positive_curve_up_to(
                taus,
                data[:, 2],
                original_effective_cutoff_b_to_a,
                method=integration_method,
            ),
            "smoothed_pc_a_to_pc_b_cutoff_found": smoothed_cutoff_found_a_to_b,
            "smoothed_pc_b_to_pc_a_cutoff_found": smoothed_cutoff_found_b_to_a,
            "smoothed_pc_a_to_pc_b_cutoff_ns": smoothed_effective_cutoff_a_to_b,
            "smoothed_pc_b_to_pc_a_cutoff_ns": smoothed_effective_cutoff_b_to_a,
            "smoothed_pc_a_to_pc_b_area_to_cutoff": integrate_positive_curve_up_to(
                smooth_a_to_b[:, 0],
                smooth_a_to_b[:, 1],
                smoothed_effective_cutoff_a_to_b,
                method=integration_method,
            ),
            "smoothed_pc_b_to_pc_a_area_to_cutoff": integrate_positive_curve_up_to(
                smooth_b_to_a[:, 0],
                smooth_b_to_a[:, 1],
                smoothed_effective_cutoff_b_to_a,
                method=integration_method,
            ),
        },
        {
            "pair": f"{metadata['pc_a']}-{metadata['pc_b']}",
            "pc_a": metadata["pc_a"],
            "pc_b": metadata["pc_b"],
            "tau_start_ns": float(taus.min()),
            "tau_stop_ns": float(taus.max()),
            "original_pc_a_to_pc_b_area": integrate_positive_curve_with_method(
                taus, original_low_a_to_b, method=integration_method
            ),
            "original_pc_b_to_pc_a_area": integrate_positive_curve_with_method(
                taus, original_low_b_to_a, method=integration_method
            ),
            "smoothed_pc_a_to_pc_b_area": integrate_positive_curve_with_method(
                smooth_a_to_b[:, 0], smoothed_low_a_to_b, method=integration_method
            ),
            "smoothed_pc_b_to_pc_a_area": integrate_positive_curve_with_method(
                smooth_b_to_a[:, 0], smoothed_low_b_to_a, method=integration_method
            ),
            "original_pc_a_to_pc_b_positive_points": int((original_low_a_to_b > 0.0).sum()),
            "original_pc_b_to_pc_a_positive_points": int((original_low_b_to_a > 0.0).sum()),
            "smoothed_pc_a_to_pc_b_positive_points": int((smoothed_low_a_to_b > 0.0).sum()),
            "smoothed_pc_b_to_pc_a_positive_points": int((smoothed_low_b_to_a > 0.0).sum()),
            "original_n_points": int(len(taus)),
            "smoothed_n_points": int(len(smooth_a_to_b)),
        },
        {
            "pair": f"{metadata['pc_a']}-{metadata['pc_b']}",
            "pc_a": metadata["pc_a"],
            "pc_b": metadata["pc_b"],
            "tau_start_ns": float(taus.min()),
            "tau_stop_ns": float(taus.max()),
            "original_pc_a_to_pc_b_area": integrate_positive_curve_with_method(
                taus, original_high_a_to_b, method=integration_method
            ),
            "original_pc_b_to_pc_a_area": integrate_positive_curve_with_method(
                taus, original_high_b_to_a, method=integration_method
            ),
            "smoothed_pc_a_to_pc_b_area": integrate_positive_curve_with_method(
                smooth_a_to_b[:, 0], smoothed_high_a_to_b, method=integration_method
            ),
            "smoothed_pc_b_to_pc_a_area": integrate_positive_curve_with_method(
                smooth_b_to_a[:, 0], smoothed_high_b_to_a, method=integration_method
            ),
            "original_pc_a_to_pc_b_positive_points": int((original_high_a_to_b > 0.0).sum()),
            "original_pc_b_to_pc_a_positive_points": int((original_high_b_to_a > 0.0).sum()),
            "smoothed_pc_a_to_pc_b_positive_points": int((smoothed_high_a_to_b > 0.0).sum()),
            "smoothed_pc_b_to_pc_a_positive_points": int((smoothed_high_b_to_a > 0.0).sum()),
            "original_n_points": int(len(taus)),
            "smoothed_n_points": int(len(smooth_a_to_b)),
        },
    )


def write_csv(output_path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_main_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "pair",
        "pc_a",
        "pc_b",
        "tau_start_ns",
        "tau_stop_ns",
        "original_pc_a_to_pc_b_area",
        "original_pc_b_to_pc_a_area",
        "smoothed_pc_a_to_pc_b_area",
        "smoothed_pc_b_to_pc_a_area",
        "original_pc_a_to_pc_b_positive_points",
        "original_pc_b_to_pc_a_positive_points",
        "smoothed_pc_a_to_pc_b_positive_points",
        "smoothed_pc_b_to_pc_a_positive_points",
        "original_n_points",
        "smoothed_n_points",
    ]
    write_csv(output_path, rows, fieldnames)


def write_cutoff_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "pair",
        "pc_a",
        "pc_b",
        "tau_start_ns",
        "tau_stop_ns",
        "original_pc_a_to_pc_b_cutoff_found",
        "original_pc_b_to_pc_a_cutoff_found",
        "original_pc_a_to_pc_b_cutoff_ns",
        "original_pc_b_to_pc_a_cutoff_ns",
        "original_pc_a_to_pc_b_area_to_cutoff",
        "original_pc_b_to_pc_a_area_to_cutoff",
        "smoothed_pc_a_to_pc_b_cutoff_found",
        "smoothed_pc_b_to_pc_a_cutoff_found",
        "smoothed_pc_a_to_pc_b_cutoff_ns",
        "smoothed_pc_b_to_pc_a_cutoff_ns",
        "smoothed_pc_a_to_pc_b_area_to_cutoff",
        "smoothed_pc_b_to_pc_a_area_to_cutoff",
    ]
    write_csv(output_path, rows, fieldnames)


def write_json(
    output_path: Path,
    rows: list[dict[str, object]],
    input_dir: Path,
    pattern: str,
    num_interp_points: int,
    smooth_window_length: int,
    smooth_poly_deg: int,
    num_passes: int,
    interpolating_func: str,
    integration_method: str,
    extra_metadata: dict[str, object] | None = None,
) -> None:
    payload = {
        "input_dir": str(input_dir),
        "pattern": pattern,
        "integration_rule": "Area under each IG curve after zeroing all IG values <= 0. Error bands are ignored.",
        "integration_method": integration_method,
        "integration_method_detail": ("Numerical quadrature applied to the clipped curve max(IG, 0)."),
        "smoothing_parameters": {
            "interp_points": num_interp_points,
            "window_length": smooth_window_length,
            "polyorder": smooth_poly_deg,
            "passes": num_passes,
            "interpolator": interpolating_func,
        },
        "rows": rows,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate pairwise IG curves from notebook-style " "IG-pc*_vs_pc*.txt files and their smoothed counterparts."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=("analysis_outputs/ig_results/" "200to490us_sklearn_standard_top5/E=1_k=50_dci=0"),
        help="Directory containing notebook-style IG-pc*_vs_pc*.txt files.",
    )
    parser.add_argument(
        "--pattern",
        default="IG-pc*_vs_pc*.txt",
        help="Glob pattern used to find input files inside input_dir.",
    )
    parser.add_argument(
        "--csv-name",
        default="ig_curve_integrals.csv",
        help="Filename for the tabular CSV output written inside input_dir.",
    )
    parser.add_argument(
        "--json-name",
        default="ig_curve_integrals.json",
        help="Filename for the JSON summary output written inside input_dir.",
    )
    parser.add_argument(
        "--cutoff-csv-name",
        default="ig_curve_integrals_cutoff.csv",
        help="Filename for the cutoff-only CSV output written inside input_dir.",
    )
    parser.add_argument(
        "--cutoff-json-name",
        default="ig_curve_integrals_cutoff.json",
        help="Filename for the cutoff-only JSON output written inside input_dir.",
    )
    parser.add_argument(
        "--low-csv-name",
        default="ig_curve_integrals_low.csv",
        help="Filename for the lower-bound full-range CSV output written inside input_dir.",
    )
    parser.add_argument(
        "--low-json-name",
        default="ig_curve_integrals_low.json",
        help="Filename for the lower-bound full-range JSON output written inside input_dir.",
    )
    parser.add_argument(
        "--high-csv-name",
        default="ig_curve_integrals_high.csv",
        help="Filename for the upper-bound full-range CSV output written inside input_dir.",
    )
    parser.add_argument(
        "--high-json-name",
        default="ig_curve_integrals_high.json",
        help="Filename for the upper-bound full-range JSON output written inside input_dir.",
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
    parser.add_argument(
        "--integration-method",
        choices=["trapezoid", "simpson"],
        default="simpson",
        help="Quadrature rule used for the IG curve integrals.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    ig_paths = sorted(input_dir.glob(args.pattern))
    if not ig_paths:
        raise SystemExit(f"No files matched pattern {args.pattern!r} in directory {input_dir}.")

    main_rows = []
    cutoff_rows = []
    low_rows = []
    high_rows = []
    for ig_path in ig_paths:
        main_row, cutoff_row, low_row, high_row = integrate_pair(
            input_path=ig_path,
            num_interp_points=args.interp_points,
            smooth_window_length=args.window_length,
            smooth_poly_deg=args.polyorder,
            num_passes=args.passes,
            interpolating_func=args.interpolator,
            integration_method=args.integration_method,
        )
        main_rows.append(main_row)
        cutoff_rows.append(cutoff_row)
        low_rows.append(low_row)
        high_rows.append(high_row)

    csv_path = input_dir / args.csv_name
    json_path = input_dir / args.json_name
    cutoff_csv_path = input_dir / args.cutoff_csv_name
    cutoff_json_path = input_dir / args.cutoff_json_name
    low_csv_path = input_dir / args.low_csv_name
    low_json_path = input_dir / args.low_json_name
    high_csv_path = input_dir / args.high_csv_name
    high_json_path = input_dir / args.high_json_name
    write_main_csv(csv_path, main_rows)
    write_cutoff_csv(cutoff_csv_path, cutoff_rows)
    write_main_csv(low_csv_path, low_rows)
    write_main_csv(high_csv_path, high_rows)
    write_json(
        json_path,
        main_rows,
        input_dir=input_dir,
        pattern=args.pattern,
        num_interp_points=args.interp_points,
        smooth_window_length=args.window_length,
        smooth_poly_deg=args.polyorder,
        num_passes=args.passes,
        interpolating_func=args.interpolator,
        integration_method=args.integration_method,
    )
    write_json(
        cutoff_json_path,
        cutoff_rows,
        input_dir=input_dir,
        pattern=args.pattern,
        num_interp_points=args.interp_points,
        smooth_window_length=args.window_length,
        smooth_poly_deg=args.polyorder,
        num_passes=args.passes,
        interpolating_func=args.interpolator,
        integration_method=args.integration_method,
        extra_metadata={
            "cutoff_rule": (
                "For each direction, compute an area truncated at the first post-onset time where "
                "IG - error becomes <= 0. The crossing time is estimated linearly between samples."
            )
        },
    )
    write_json(
        low_json_path,
        low_rows,
        input_dir=input_dir,
        pattern=args.pattern,
        num_interp_points=args.interp_points,
        smooth_window_length=args.window_length,
        smooth_poly_deg=args.polyorder,
        num_passes=args.passes,
        interpolating_func=args.interpolator,
        integration_method=args.integration_method,
        extra_metadata={
            "integration_rule": (
                "Area under each lower-bound IG curve after zeroing all values <= 0. " "The lower-bound curve is IG - error."
            ),
            "bound_type": "low",
            "bound_detail": "Lower-bound full-range integration using IG - error.",
        },
    )
    write_json(
        high_json_path,
        high_rows,
        input_dir=input_dir,
        pattern=args.pattern,
        num_interp_points=args.interp_points,
        smooth_window_length=args.window_length,
        smooth_poly_deg=args.polyorder,
        num_passes=args.passes,
        interpolating_func=args.interpolator,
        integration_method=args.integration_method,
        extra_metadata={
            "integration_rule": (
                "Area under each upper-bound IG curve after zeroing all values <= 0. " "The upper-bound curve is IG + error."
            ),
            "bound_type": "high",
            "bound_detail": "Upper-bound full-range integration using IG + error.",
        },
    )

    print(csv_path)
    print(json_path)
    print(cutoff_csv_path)
    print(cutoff_json_path)
    print(low_csv_path)
    print(low_json_path)
    print(high_csv_path)
    print(high_json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
