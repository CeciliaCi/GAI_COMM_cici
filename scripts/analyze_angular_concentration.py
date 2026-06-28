#!/usr/bin/env python3
"""Analyze angular-domain energy concentration for LEO channel datasets."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from loaders import Channels


DEFAULT_DATASETS = (
    (
        "Urban p013 test",
        ("dataset/p013/LEO_Urban_Mixed_h1000km_el15-90_path10_seed2222.mat",),
    ),
    (
        "DenseUrban p013 test",
        ("dataset/p013/LEO_DenseUrban_Mixed_h1000km_el10-90_path20_seed2222.mat",),
    ),
    (
        "Urban p006 test",
        ("dataset/p006/LEO_Urban_Mixed_h1000km_el15-90_path10_seed2222.mat",),
    ),
    (
        "DenseUrban p006 test",
        ("dataset/p006/LEO_DenseUrban_Mixed_h1000km_el10-90_path20_seed2222.mat",),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare angular-domain energy concentration of LEO channel datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs="+",
        metavar=("LABEL", "FILE"),
        help=(
            "Dataset to analyze as LABEL FILE [FILE ...]. "
            "Can be repeated. Defaults to p013/p006 Urban and DenseUrban test files."
        ),
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Top-k angular energy fractions to report.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.90, 0.95],
        help="Energy thresholds for reporting required angular bins.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional number of leading samples to analyze from each dataset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path for the summary metrics.",
    )
    parser.add_argument(
        "--no-strict-dtype",
        action="store_true",
        help="Allow non-complex128 channel arrays when loading .mat files.",
    )
    return parser.parse_args()


def resolve_files(files: list[str] | tuple[str, ...]) -> list[str]:
    resolved = []
    for file_name in files:
        path = Path(file_name)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        resolved.append(str(path.resolve()))
    return resolved


def configured_datasets(args: argparse.Namespace) -> list[tuple[str, tuple[str, ...]]]:
    if not args.dataset:
        return list(DEFAULT_DATASETS)
    datasets = []
    for item in args.dataset:
        if len(item) < 2:
            raise ValueError("--dataset requires LABEL followed by at least one FILE.")
        datasets.append((item[0], tuple(item[1:])))
    return datasets


def summarize_dataset(
    *,
    label: str,
    files: tuple[str, ...],
    top_k: list[int],
    thresholds: list[float],
    max_samples: int | None,
    strict_dtype: bool,
) -> dict[str, float | int | str]:
    channels = Channels(files=resolve_files(files), strict_dtype=strict_dtype)
    h = channels.channels[:max_samples] if max_samples is not None else channels.channels
    angular = np.fft.fft2(h, axes=(-2, -1), norm="ortho")
    energy = np.abs(angular).reshape(angular.shape[0], -1) ** 2
    total_energy = np.maximum(energy.sum(axis=1, keepdims=True), 1e-300)
    fractions = energy / total_energy
    sorted_fractions = np.sort(fractions, axis=1)[:, ::-1]
    cdf = np.cumsum(sorted_fractions, axis=1)

    row: dict[str, float | int | str] = {
        "label": label,
        "files": ";".join(files),
        "num_samples": int(h.shape[0]),
        "num_angular_bins": int(energy.shape[1]),
    }
    for k in top_k:
        if k <= 0:
            raise ValueError("--top-k values must be positive.")
        top_fraction = sorted_fractions[:, : min(k, sorted_fractions.shape[1])].sum(axis=1)
        row[f"top{k}_mean"] = float(np.mean(top_fraction))
        row[f"top{k}_median"] = float(np.median(top_fraction))

    effective_bins = 1.0 / np.sum(fractions**2, axis=1)
    row["effective_bins_mean"] = float(np.mean(effective_bins))
    row["effective_bins_median"] = float(np.median(effective_bins))

    for threshold in thresholds:
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError("--thresholds values must be in (0, 1].")
        bins_needed = np.argmax(cdf >= threshold, axis=1) + 1
        threshold_label = f"{int(round(threshold * 100))}"
        row[f"bins{threshold_label}_mean"] = float(np.mean(bins_needed))
        row[f"bins{threshold_label}_median"] = float(np.median(bins_needed))

    return row


def print_summary(row: dict[str, float | int | str], top_k: list[int], thresholds: list[float]) -> None:
    print(row["label"])
    print(f"  samples: {row['num_samples']} bins: {row['num_angular_bins']}")
    top_means = " ".join(f"top{k}={row[f'top{k}_mean']:.4f}" for k in top_k)
    top_medians = " ".join(f"top{k}={row[f'top{k}_median']:.4f}" for k in top_k)
    print(f"  top-k mean: {top_means}")
    print(f"  top-k median: {top_medians}")
    print(
        "  effective bins mean/median: "
        f"{row['effective_bins_mean']:.2f} {row['effective_bins_median']:.2f}"
    )
    for threshold in thresholds:
        threshold_label = f"{int(round(threshold * 100))}"
        print(
            f"  bins for {threshold_label}% mean/median: "
            f"{row[f'bins{threshold_label}_mean']:.2f} {row[f'bins{threshold_label}_median']:.2f}"
        )


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    datasets = configured_datasets(args)
    top_k = sorted(set(args.top_k))
    thresholds = sorted(set(args.thresholds))

    rows = []
    for label, files in datasets:
        row = summarize_dataset(
            label=label,
            files=files,
            top_k=top_k,
            thresholds=thresholds,
            max_samples=args.max_samples,
            strict_dtype=not args.no_strict_dtype,
        )
        rows.append(row)
        print_summary(row, top_k, thresholds)

    if args.output:
        output = Path(args.output).resolve()
        write_csv(output, rows)
        print(f"Saved angular concentration summary: {output}")


if __name__ == "__main__":
    main()
