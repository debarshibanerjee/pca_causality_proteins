#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex", "std-colors"])
    except Exception:
        plt.style.use("default")
    plt.rcParams["font.size"] = 16
    plt.rcParams["figure.constrained_layout.use"] = True


def as_component_array(components: np.ndarray) -> np.ndarray:
    components = np.asarray(components, dtype=np.float64)
    if components.ndim == 3:
        if components.shape[2] != 3:
            raise ValueError(f"Expected component shape (n_pcs, n_ca, 3), got {components.shape}")
        return components

    if components.ndim == 2:
        if components.shape[1] % 3 != 0:
            raise ValueError(f"Flat component vectors must have length 3*N, got {components.shape}")
        return components.reshape(components.shape[0], components.shape[1] // 3, 3)

    raise ValueError(f"Expected 2D or 3D components, got shape {components.shape}")


def parse_ca_residues(
    topology_path: Path,
    n_ca: int,
    residue_start: int | None = None,
    residue_stop: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with topology_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            residue_name = line[17:20].strip()
            chain_id = line[21].strip()
            residue_id = line[22:26].strip()
            insertion_code = line[26].strip()
            residue_number = int(residue_id)
            if residue_start is not None and residue_number < residue_start:
                continue
            if residue_stop is not None and residue_number > residue_stop:
                continue
            residue_label = f"{residue_name}{residue_id}{insertion_code}"
            if chain_id:
                residue_label = f"{residue_label}:{chain_id}"
            rows.append(
                {
                    "ca_index": len(rows) + 1,
                    "residue_name": residue_name,
                    "chain_id": chain_id,
                    "residue_id": residue_id,
                    "residue_number": residue_number,
                    "insertion_code": insertion_code,
                    "residue_label": residue_label,
                }
            )

    if len(rows) != n_ca:
        raise ValueError(f"Topology contains {len(rows)} CA atoms, but components contain {n_ca} CA atoms")
    return pd.DataFrame(rows)


def fallback_residues(n_ca: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ca_index": np.arange(1, n_ca + 1, dtype=int),
            "residue_name": [""] * n_ca,
            "chain_id": [""] * n_ca,
            "residue_id": [str(i) for i in range(1, n_ca + 1)],
            "insertion_code": [""] * n_ca,
            "residue_label": [str(i) for i in range(1, n_ca + 1)],
        }
    )


def bruschweiler_collectivity(components: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    amplitudes = np.sum(components * components, axis=2)
    totals = amplitudes.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        bad_modes = np.where(totals[:, 0] <= 0.0)[0] + 1
        raise ValueError(f"Zero-norm component vectors for PCs: {bad_modes.tolist()}")

    contributions = amplitudes / totals
    entropy_terms = np.where(contributions > eps, contributions * np.log(contributions), 0.0)
    entropy = -np.sum(entropy_terms, axis=1)
    n_eff = np.exp(entropy)
    kappa = n_eff / components.shape[1]

    return {
        "amplitudes": amplitudes,
        "contributions": contributions,
        "entropy": entropy,
        "n_eff": n_eff,
        "kappa": kappa,
    }


def load_spectrum(path: Path | None, n_pcs: int) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame({"component": np.arange(1, n_pcs + 1, dtype=int)})
    table = pd.read_csv(path)
    if "component" not in table.columns:
        raise ValueError(f"Spectrum file {path} must contain a 'component' column")
    return table


def top_residue_summary(contributions: np.ndarray, residue_table: pd.DataFrame, top_n: int) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    values: list[str] = []
    residue_labels = residue_table["residue_label"].astype(str).to_numpy()
    for row in contributions:
        order = np.argsort(row)[::-1][:top_n]
        labels.append(";".join(residue_labels[order]))
        values.append(";".join(f"{row[index]:.6g}" for index in order))
    return labels, values


def build_collectivity_table(
    results: dict[str, np.ndarray],
    spectrum: pd.DataFrame,
    residue_table: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    n_pcs = results["kappa"].size
    table = pd.DataFrame(
        {
            "component": np.arange(1, n_pcs + 1, dtype=int),
            "kappa": results["kappa"],
            "n_eff": results["n_eff"],
            "entropy": results["entropy"],
            "n_ca": np.full(n_pcs, len(residue_table), dtype=int),
        }
    )
    top_labels, top_values = top_residue_summary(results["contributions"], residue_table, top_n=top_n)
    table["top_residues"] = top_labels
    table["top_contributions"] = top_values

    spectrum_columns = [
        column
        for column in ["component", "eigenvalue", "explained_variance_ratio", "cumulative_explained_variance_ratio"]
        if column in spectrum.columns
    ]
    if len(spectrum_columns) > 1:
        table = table.merge(spectrum[spectrum_columns], on="component", how="left")
        ordered = [
            "component",
            "eigenvalue",
            "explained_variance_ratio",
            "cumulative_explained_variance_ratio",
            "kappa",
            "n_eff",
            "entropy",
            "n_ca",
            "top_residues",
            "top_contributions",
        ]
        table = table[[column for column in ordered if column in table.columns]]

    return table


def build_residue_contribution_table(contributions: np.ndarray, residue_table: pd.DataFrame) -> pd.DataFrame:
    table = residue_table.copy()
    for pc_index in range(contributions.shape[0]):
        table[f"pc{pc_index + 1:02d}_contribution"] = contributions[pc_index]
    return table


def parse_pc_selection(raw_value: str, n_pcs: int) -> list[int]:
    value = raw_value.strip().lower()
    if value == "none":
        return []
    if value == "all":
        return list(range(1, n_pcs + 1))

    selected: list[int] = []
    for token in re.split(r"[,\s]+", value):
        if not token:
            continue
        if "-" in token:
            start_raw, stop_raw = token.split("-", 1)
            start = int(start_raw)
            stop = int(stop_raw)
            selected.extend(range(start, stop + 1))
        else:
            selected.append(int(token))

    unique = sorted(set(selected))
    invalid = [pc for pc in unique if pc < 1 or pc > n_pcs]
    if invalid:
        raise ValueError(f"Selected PCs are outside 1-{n_pcs}: {invalid}")
    return unique


def save_collectivity_plot(table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(table["component"], table["kappa"], marker="o", linewidth=2.0, color="#0f4c5c")
    ax.set_xlabel("Principal component")
    ax.set_ylabel(r"Bruschweiler collectivity $\kappa$")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(table["component"])
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_neff_plot(table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(table["component"], table["n_eff"], marker="o", linewidth=2.0, color="#9a031e")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Effective participating residues")
    ax.set_xticks(table["component"])
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_residue_plot(
    pc_number: int,
    contributions: np.ndarray,
    residue_table: pd.DataFrame,
    output_path: Path,
    top_n: int,
) -> None:
    residue_indices = residue_table["ca_index"].to_numpy(dtype=int)
    residue_labels = residue_table["residue_label"].astype(str).to_numpy()
    values = contributions[pc_number - 1]
    top_indices = np.argsort(values)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(residue_indices, values, marker="o", markersize=4.0, linewidth=1.8, color="#16324f")
    ax.scatter(residue_indices[top_indices], values[top_indices], s=42, color="#9a031e", zorder=3)
    for index in top_indices:
        ax.annotate(
            residue_labels[index],
            (residue_indices[index], values[index]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            rotation=45,
        )
    ax.set_xlabel("Residue index")
    ax.set_ylabel(r"Normalized mode contribution $p_i$")
    ax.set_title(f"PC{pc_number} residue-wise collectivity contributions")
    ax.set_xlim(residue_indices.min() - 1, residue_indices.max() + 1)
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the Bruschweiler collectivity index for saved C-alpha PCA component vectors."
    )
    parser.add_argument(
        "--components",
        default="analysis_outputs/pca_sklearn_standard/r2-70_650us_sklearn_standard_components.npy",
        help="Saved PCA component array, shape (n_pcs, n_ca, 3) or (n_pcs, 3*n_ca).",
    )
    parser.add_argument(
        "--spectrum",
        default="analysis_outputs/pca_sklearn_standard/r2-70_650us_sklearn_standard_spectrum.csv",
        help="Optional PCA spectrum CSV to merge into the collectivity table. Use 'none' to disable.",
    )
    parser.add_argument(
        "--topology",
        default="analysis_outputs/topology/protein_ca.pdb",
        help="CA topology PDB used to attach residue labels. Use 'none' to label residues by index only.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/collectivity/r2-70_650us_sklearn_standard",
        help="Directory for collectivity tables and plots.",
    )
    parser.add_argument("--label", default="r2-70_650us_sklearn_standard")
    parser.add_argument("--residue-start", type=int, default=2)
    parser.add_argument("--residue-stop", type=int, default=70)
    parser.add_argument("--n-pcs", type=int, default=10, help="Number of leading PCs to analyze. Defaults to all.")
    parser.add_argument("--top-residues", type=int, default=10, help="Number of largest residue contributors to list.")
    parser.add_argument(
        "--top-summary-pcs",
        type=int,
        default=5,
        help="Also write summary collectivity plots restricted to this many leading PCs. Use 0 to disable.",
    )
    parser.add_argument(
        "--residue-plot-pcs",
        default="all",
        help="PCs for residue-wise plots, e.g. 'all', 'none', '1-5', or '1,3,6'.",
    )
    parser.add_argument("--eps", type=float, default=1.0e-15, help="Threshold for omitting zero p*log(p) terms.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_residues < 1:
        raise ValueError("--top-residues must be positive")
    if args.eps < 0.0:
        raise ValueError("--eps must be non-negative")
    if args.top_summary_pcs < 0:
        raise ValueError("--top-summary-pcs must be non-negative")
    if args.residue_stop < args.residue_start:
        raise ValueError("--residue-stop must be greater than or equal to --residue-start")

    apply_plot_style()
    output_dir = ensure_dir(args.output_dir)

    components = as_component_array(np.load(args.components))
    if args.n_pcs is not None:
        if args.n_pcs < 1:
            raise ValueError("--n-pcs must be positive")
        if args.n_pcs > components.shape[0]:
            raise ValueError(f"Requested {args.n_pcs} PCs but components only contain {components.shape[0]}")
        components = components[: args.n_pcs]

    n_pcs, n_ca, _ = components.shape
    topology_path = None if args.topology.lower() == "none" else Path(args.topology)
    if topology_path is None:
        residue_table = fallback_residues(n_ca)
    else:
        residue_table = parse_ca_residues(topology_path, n_ca, args.residue_start, args.residue_stop)

    spectrum_path = None if args.spectrum.lower() == "none" else Path(args.spectrum)
    spectrum = load_spectrum(spectrum_path, n_pcs)
    results = bruschweiler_collectivity(components, eps=args.eps)

    collectivity_table = build_collectivity_table(
        results=results,
        spectrum=spectrum,
        residue_table=residue_table,
        top_n=args.top_residues,
    )
    residue_contribution_table = build_residue_contribution_table(results["contributions"], residue_table)

    collectivity_csv = output_dir / f"{args.label}_collectivity.csv"
    collectivity_json = output_dir / f"{args.label}_collectivity.json"
    contributions_npy = output_dir / f"{args.label}_residue_contributions.npy"
    contributions_csv = output_dir / f"{args.label}_residue_contributions.csv"
    collectivity_plot = output_dir / f"{args.label}_collectivity.png"
    neff_plot = output_dir / f"{args.label}_n_eff.png"
    top_collectivity_plot = None
    top_neff_plot = None

    collectivity_table.to_csv(collectivity_csv, index=False)
    residue_contribution_table.to_csv(contributions_csv, index=False)
    np.save(contributions_npy, results["contributions"])
    save_collectivity_plot(collectivity_table, collectivity_plot)
    save_neff_plot(collectivity_table, neff_plot)
    if args.top_summary_pcs > 0:
        top_n = min(args.top_summary_pcs, len(collectivity_table))
        top_table = collectivity_table.iloc[:top_n].copy()
        top_collectivity_plot = output_dir / f"{args.label}_top{top_n}_collectivity.png"
        top_neff_plot = output_dir / f"{args.label}_top{top_n}_n_eff.png"
        save_collectivity_plot(top_table, top_collectivity_plot)
        save_neff_plot(top_table, top_neff_plot)

    residue_plot_pcs = parse_pc_selection(args.residue_plot_pcs, n_pcs)
    residue_plot_files = []
    for pc_number in residue_plot_pcs:
        output_path = output_dir / f"{args.label}_pc{pc_number:02d}_residue_contributions.png"
        save_residue_plot(
            pc_number=pc_number,
            contributions=results["contributions"],
            residue_table=residue_table,
            output_path=output_path,
            top_n=args.top_residues,
        )
        residue_plot_files.append(str(output_path))

    payload = {
        "components_file": args.components,
        "spectrum_file": str(spectrum_path) if spectrum_path is not None else None,
        "topology_file": str(topology_path) if topology_path is not None else None,
        "residue_start": int(args.residue_start),
        "residue_stop": int(args.residue_stop),
        "label": args.label,
        "n_pcs": int(n_pcs),
        "n_ca": int(n_ca),
        "eps": float(args.eps),
        "definition": "kappa = exp(-sum_i p_i log p_i) / N, where p_i is the normalized squared CA displacement amplitude.",
        "output_files": {
            "collectivity_csv": str(collectivity_csv),
            "collectivity_json": str(collectivity_json),
            "residue_contributions_npy": str(contributions_npy),
            "residue_contributions_csv": str(contributions_csv),
            "collectivity_plot": str(collectivity_plot),
            "n_eff_plot": str(neff_plot),
            "top_collectivity_plot": str(top_collectivity_plot) if top_collectivity_plot is not None else None,
            "top_n_eff_plot": str(top_neff_plot) if top_neff_plot is not None else None,
            "residue_plots": residue_plot_files,
        },
        "collectivity": collectivity_table.to_dict(orient="records"),
    }
    collectivity_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote collectivity table to {collectivity_csv}")
    print(f"Wrote residue contributions to {contributions_csv}")
    print(f"Wrote {len(residue_plot_files)} residue-wise contribution plots")


if __name__ == "__main__":
    main()
