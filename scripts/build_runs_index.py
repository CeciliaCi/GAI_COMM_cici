#!/usr/bin/env python3
"""Rebuild results/runs_index.csv from existing training result directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modules import result_io


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan existing DMCE training result directories and rebuild runs_index.csv."
    )
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_DIR / "results"),
        help="Root results directory. Defaults to ./results.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <results-root>/runs_index.csv.",
    )
    parser.add_argument(
        "--include-best-models",
        action="store_true",
        help="Also scan results/best_models_dm_paper/*/sim_params.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered run count without writing the CSV.",
    )
    return parser.parse_args()


def discover_run_dirs(results_root: Path, *, include_best_models: bool) -> list[Path]:
    run_dirs = [
        child
        for child in results_root.iterdir()
        if child.is_dir() and (child / "sim_params.json").is_file()
    ]
    if include_best_models:
        best_root = results_root / "best_models_dm_paper"
        if best_root.is_dir():
            run_dirs.extend(
                child
                for child in best_root.iterdir()
                if child.is_dir() and (child / "sim_params.json").is_file()
            )
    return sorted(run_dirs, key=lambda item: str(item))


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output = Path(args.output).resolve() if args.output else results_root / "runs_index.csv"
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    run_dirs = discover_run_dirs(results_root, include_best_models=args.include_best_models)
    rows = [
        result_io.existing_run_index_row(run_dir, cwd=PROJECT_DIR, results_root=results_root)
        for run_dir in run_dirs
    ]
    if args.dry_run:
        print(f"Discovered {len(rows)} run(s).")
        for row in rows:
            print(row["run_dir"])
        return

    result_io.write_run_index(output, rows)
    print(f"Wrote {len(rows)} run(s) to {output}")


if __name__ == "__main__":
    main()
