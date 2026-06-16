#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy import interpolate
from scipy.signal import savgol_filter


IG_FILENAME_RE = re.compile(r"IG-(pc\d+)_vs_(pc\d+)_E-(?P<E>[^_]+)_k-(?P<k>[^_]+)_dci-(?P<dci>[^.]+)\.txt$")
NOTEBOOK_IG_FILENAME_RE = re.compile(r"IG-(pc\d+)_vs_(pc\d+)\.txt$")

DEFAULT_INTERP_POINTS = 1001
DEFAULT_WINDOW_LENGTH = 15
DEFAULT_POLYORDER = 3
DEFAULT_PASSES = 13
DEFAULT_INTERPOLATOR = "cubic"


def load_ig_table(input_path: Path) -> np.ndarray:
    data = np.loadtxt(input_path)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"{input_path} must be a 2D text table with at least 7 columns; got shape {data.shape}.")
    return data


def parse_pair_metadata(path: Path) -> dict[str, str]:
    match = IG_FILENAME_RE.match(path.name)
    if match is not None:
        data = match.groupdict()
        data["pc_a"] = match.group(1)
        data["pc_b"] = match.group(2)
        return data

    notebook_match = NOTEBOOK_IG_FILENAME_RE.match(path.name)
    if notebook_match is not None:
        return {
            "pc_a": notebook_match.group(1),
            "pc_b": notebook_match.group(2),
        }

    raise ValueError(f"Unrecognized IG filename format: {path.name}")


def build_smoothed_output_path(input_path: Path, output_suffix: str) -> Path:
    metadata = parse_pair_metadata(input_path)
    pair_label = f"{metadata['pc_a']}-vs-{metadata['pc_b']}"
    return input_path.with_name(f"IG_{pair_label}{output_suffix}.png")


def find_closest_indices(continuous_times: np.ndarray, significant_times: np.ndarray) -> np.ndarray:
    """Map discrete significant tau values onto the interpolated x-grid."""
    if significant_times.size == 0:
        return np.array([], dtype=int)
    return np.array(
        [(np.abs(continuous_times - time)).argmin() for time in significant_times],
        dtype=int,
    )


def adjusted_window_length(length: int, requested: int, polyorder: int) -> int:
    """Return a valid odd Savitzky-Golay window for the given series length."""
    if length <= polyorder + 1:
        raise ValueError(f"Need at least {polyorder + 2} points for polyorder={polyorder}, got {length}.")

    window = min(requested, length if length % 2 == 1 else length - 1)
    if window <= polyorder:
        window = polyorder + 1
    if window % 2 == 0:
        window += 1
    if window > length:
        window -= 2
    if window <= polyorder:
        raise ValueError(f"Could not construct a valid odd window for length={length}, polyorder={polyorder}.")
    return window


def smooth_and_interpolate(
    data: np.ndarray,
    num_interp_points: int = DEFAULT_INTERP_POINTS,
    smooth_window_length: int = DEFAULT_WINDOW_LENGTH,
    smooth_poly_deg: int = DEFAULT_POLYORDER,
    num_passes: int = DEFAULT_PASSES,
    interpolating_func: str = DEFAULT_INTERPOLATOR,
) -> np.ndarray:
    """Smooth a 2-column x/y array, then interpolate it on a denser x-grid."""
    x_values = data[:, 0]
    y_values = np.array(data[:, 1], copy=True)

    window_length = adjusted_window_length(length=len(data), requested=smooth_window_length, polyorder=smooth_poly_deg)

    for _ in range(num_passes):
        y_values = savgol_filter(y_values, window_length, smooth_poly_deg)

    interpolator = interpolate.interp1d(x_values, y_values, kind=interpolating_func)
    x_interp = np.linspace(x_values.min(), x_values.max(), num_interp_points)
    y_interp = interpolator(x_interp)
    return np.column_stack([x_interp, y_interp])


def integrate_positive_curve(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Integrate only the positive part of a curve by zeroing non-positive samples."""
    positive_y = np.clip(y_values, a_min=0.0, a_max=None)
    return integrate_curve(x_values=x_values, y_values=positive_y, method="trapezoid")


def integrate_curve(x_values: np.ndarray, y_values: np.ndarray, method: str = "trapezoid") -> float:
    """Integrate a curve using the requested quadrature rule."""
    if method == "trapezoid":
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y_values, x_values))
        return float(np.trapz(y_values, x_values))
    if method == "simpson":
        return float(simpson(y_values, x=x_values))
    raise ValueError(f"Unsupported integration method: {method}")


def first_nonpositive_lower_bound_time(x_values: np.ndarray, y_values: np.ndarray, error_values: np.ndarray) -> float | None:
    """
    Find the first post-onset time where y - error reaches a non-positive value.

    The search starts only after the lower bound has been strictly positive at least once.
    This avoids the trivial tau=0 cutoff when the curve starts at exactly zero.
    A linear interpolation is used to estimate the zero-crossing time between samples.
    """
    lower_bound = y_values - error_values
    positive_indices = np.flatnonzero(lower_bound > 0.0)
    if positive_indices.size == 0:
        return None

    start_index = int(positive_indices[0])
    for idx in range(start_index + 1, len(x_values)):
        previous_lower = lower_bound[idx - 1]
        current_lower = lower_bound[idx]
        if current_lower > 0.0:
            continue
        x0 = float(x_values[idx - 1])
        x1 = float(x_values[idx])
        if previous_lower <= 0.0 or x1 == x0:
            return x1
        fraction = previous_lower / (previous_lower - current_lower)
        return x0 + fraction * (x1 - x0)

    return None


def integrate_positive_curve_up_to(
    x_values: np.ndarray,
    y_values: np.ndarray,
    stop_time: float | None,
    method: str = "trapezoid",
) -> float:
    """Integrate the positive part of a curve up to a given stop time."""
    if stop_time is None:
        return integrate_positive_curve_with_method(x_values, y_values, method=method)

    if stop_time <= float(x_values[0]):
        return 0.0

    if stop_time >= float(x_values[-1]):
        return integrate_positive_curve_with_method(x_values, y_values, method=method)

    insert_index = int(np.searchsorted(x_values, stop_time, side="right"))
    x_segment = np.concatenate([x_values[:insert_index], np.array([stop_time])])
    y_stop = np.interp(stop_time, x_values, y_values)
    y_segment = np.concatenate([y_values[:insert_index], np.array([y_stop])])
    return integrate_positive_curve_with_method(x_segment, y_segment, method=method)


def integrate_positive_curve_with_method(x_values: np.ndarray, y_values: np.ndarray, method: str = "trapezoid") -> float:
    """Integrate only the positive part of a curve with a selectable rule."""
    positive_y = np.clip(y_values, a_min=0.0, a_max=None)
    return integrate_curve(x_values=x_values, y_values=positive_y, method=method)
