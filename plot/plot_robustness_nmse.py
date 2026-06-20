#!/usr/bin/env python3
"""Plot Doppler/elevation robustness curves from a trained LEO DM estimator."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from DMCE import CNN, DiffusionModel
from DMCE.utils import cmplx2real, real2cmplx
from loaders import Channels
from modules import utils as ut


DEFAULT_STRESS_SNRS = [-15, -10, -5, 0, 5, 10, 15, 20]
METHOD_ORDER = ("dm", "lmmse", "ls")
METHOD_LABELS = {
    "dm": "DM",
    "lmmse": "LMMSE",
    "ls": "LS",
}


@dataclass
class LoadedRun:
    model_dir: Path
    params: dict
    train_channels: np.ndarray
    test_channels: np.ndarray
    doppler_hz: np.ndarray
    elevation_deg: np.ndarray


@dataclass
class EvaluationResult:
    nmse: dict[str, np.ndarray]
    y_spatial: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-sample NMSE robustness against Doppler and elevation."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Result directory containing sim_params.json and train_models/. Defaults to the latest trained result.",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--snr-db", type=float, default=0.0, help="Fixed SNR for binned robustness plots.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for AWGN generation.")
    parser.add_argument("--num-bins", type=int, default=6, help="Number of Doppler/elevation bins.")
    parser.add_argument(
        "--min-bin-samples",
        type=int,
        default=5,
        help="Minimum samples required for a valid plotted bin.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size. Defaults to tester_dict.batch_size or 512.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=list(METHOD_ORDER),
        help="Estimators to evaluate.",
    )
    parser.add_argument(
        "--heatmap-value",
        choices=("dm", "dm-minus-lmmse"),
        default="dm",
        help="Doppler-elevation heatmap value.",
    )
    parser.add_argument(
        "--stress-snrs",
        nargs="+",
        type=float,
        default=DEFAULT_STRESS_SNRS,
        help="SNR grid for stress-scenario NMSE curves.",
    )
    parser.add_argument(
        "--max-stress-samples",
        type=int,
        default=128,
        help="Maximum samples per stress group and SNR.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "results" / "robustness_nmse"),
        help="Directory for generated figures and CSV files.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_arg


def latest_model_dir(results_root: Path) -> Path:
    candidates = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and (path / "sim_params.json").is_file() and (path / "train_models").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No trained DM result directory found under {results_root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_params(model_dir: Path) -> dict:
    params_path = model_dir / "sim_params.json"
    with params_path.open("r") as handle:
        return json.load(handle)


def tupleize_model_dicts(params: dict, device: str) -> tuple[dict, dict]:
    cnn_dict = dict(params["unet_dict"])
    diff_model_dict = dict(params["diff_model_dict"])
    for key in ("data_shape", "ch_layers_pre", "ch_layers_post", "kernel_size"):
        if key in cnn_dict and isinstance(cnn_dict[key], list):
            cnn_dict[key] = tuple(cnn_dict[key])
    if isinstance(diff_model_dict.get("data_shape"), list):
        diff_model_dict["data_shape"] = tuple(diff_model_dict["data_shape"])
    cnn_dict["device"] = device
    return cnn_dict, diff_model_dict


def latest_checkpoint(model_dir: Path) -> Path:
    checkpoints = sorted((model_dir / "train_models").glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoint found in {model_dir / 'train_models'}")
    return checkpoints[-1]


def resolve_files(file_paths: Iterable[str] | None) -> list[str]:
    if not file_paths:
        return []
    resolved: list[str] = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        resolved.append(str(path.resolve()))
    return resolved


def geometry_vector(record_count: int, sample_info: dict, field: str) -> np.ndarray:
    if field not in sample_info:
        raise KeyError(f"sample_info is missing required field {field!r}")
    values = np.asarray(sample_info[field]).squeeze()
    if values.ndim == 0:
        values = np.full(record_count, float(values))
    elif values.shape[0] == record_count:
        values = values.reshape(record_count, -1)[:, 0]
    elif values.shape[-1] == record_count:
        values = np.moveaxis(values, -1, 0).reshape(record_count, -1)[:, 0]
    else:
        raise ValueError(f"Field {field!r} with shape {values.shape} cannot align to {record_count} samples")
    return values.astype(float, copy=False)


def load_files_with_geometry(files: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = Channels(files=files, strict_dtype=True)
    dopplers: list[np.ndarray] = []
    elevations: list[np.ndarray] = []
    for record in channels.records:
        sample_count = record.channels.shape[0]
        dopplers.append(geometry_vector(sample_count, record.sample_info, "max_doppler_hz"))
        elevations.append(geometry_vector(sample_count, record.sample_info, "elevation_deg"))
    return channels.channels, np.concatenate(dopplers), np.concatenate(elevations)


def load_run(model_dir: Path) -> LoadedRun:
    params = load_params(model_dir)
    data_dict = params["data_dict"]
    train_files = resolve_files(data_dict.get("train_files"))
    test_files = resolve_files(data_dict.get("test_files") or data_dict.get("val_files"))
    if not train_files or not test_files:
        raise ValueError(
            "plot/plot_robustness_nmse.py requires sim_params.json with explicit train_files and test_files/val_files."
        )

    train_channels = Channels(files=train_files, strict_dtype=True).channels
    test_channels, doppler_hz, elevation_deg = load_files_with_geometry(test_files)
    power_factor = data_dict.get("power_normalization_factor")
    if power_factor:
        train_channels = train_channels / float(power_factor)
        test_channels = test_channels / float(power_factor)
    return LoadedRun(
        model_dir=model_dir,
        params=params,
        train_channels=train_channels,
        test_channels=test_channels,
        doppler_hz=doppler_hz,
        elevation_deg=elevation_deg,
    )


def load_diffusion_model(params: dict, model_dir: Path, device: str) -> DiffusionModel:
    cnn_dict, diff_model_dict = tupleize_model_dicts(params, device)
    cnn = CNN(**cnn_dict)
    diffusion_model = DiffusionModel(cnn, **diff_model_dict)
    checkpoint = torch.load(latest_checkpoint(model_dir), map_location=device)
    diffusion_model.load_state_dict(checkpoint["model"])
    diffusion_model.to(device)
    diffusion_model.eval()
    return diffusion_model


def complex_channels_to_tensor(channels: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(channels[:, None, :]))
    return cmplx2real(tensor, dim=1, new_dim=False).float()


def tensor_to_complex_numpy(tensor: torch.Tensor) -> np.ndarray:
    return real2cmplx(tensor.detach().cpu(), dim=1).numpy()


def set_noise_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_sample_nmse(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, reference.ndim))
    numerator = np.sum(np.abs(estimate - reference) ** 2, axis=axes)
    denominator = np.sum(np.abs(reference) ** 2, axis=axes)
    return numerator / np.maximum(denominator, np.finfo(np.float64).eps)


def angular_diag_lmmse(y: np.ndarray, angular_variance: np.ndarray, sigma2: float) -> np.ndarray:
    y_ang = np.fft.fft2(y, axes=(-2, -1), norm="ortho")
    h_hat_ang = (angular_variance / (angular_variance + sigma2))[None, :, :] * y_ang
    return np.fft.ifft2(h_hat_ang, axes=(-2, -1), norm="ortho")


def estimate_angular_variance(train_channels: np.ndarray) -> np.ndarray:
    train_ang = np.fft.fft2(train_channels, axes=(-2, -1), norm="ortho")
    return np.mean(np.abs(train_ang) ** 2, axis=0)


def evaluate_methods(
    *,
    channels: np.ndarray,
    snr_db: float,
    seed: int,
    methods: list[str],
    diffusion_model: DiffusionModel,
    angular_variance: np.ndarray,
    fft_pre: bool,
    mode: str,
    device: str,
    batch_size: int,
) -> EvaluationResult:
    set_noise_seed(seed)
    snr_linear = 10 ** (snr_db / 10.0)
    x_spatial = complex_channels_to_tensor(channels).to(device)
    x_model = ut.complex_1d_fft(x_spatial, ifft=False, mode=mode) if fft_pre else x_spatial
    y_model = x_model + diffusion_model.noise_multiplier / math.sqrt(snr_linear) * torch.randn_like(x_model)
    y_spatial = ut.complex_1d_fft(y_model, ifft=True, mode=mode) if fft_pre else y_model
    y_spatial_np = tensor_to_complex_numpy(y_spatial)

    nmse: dict[str, np.ndarray] = {}
    if "ls" in methods:
        nmse["ls"] = per_sample_nmse(channels, y_spatial_np)
    if "lmmse" in methods:
        sigma2 = 10 ** (-snr_db / 10.0)
        h_hat_lmmse = angular_diag_lmmse(y_spatial_np, angular_variance, sigma2)
        nmse["lmmse"] = per_sample_nmse(channels, h_hat_lmmse)
    if "dm" in methods:
        dm_estimates: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, y_model.shape[0], batch_size):
                stop = min(start + batch_size, y_model.shape[0])
                x_est_model = diffusion_model.generate_estimate(
                    y_model[start:stop],
                    snr_linear,
                    add_random=False,
                    return_all_timesteps=False,
                )
                x_est_spatial = ut.complex_1d_fft(x_est_model, ifft=True, mode=mode) if fft_pre else x_est_model
                dm_estimates.append(tensor_to_complex_numpy(x_est_spatial))
        h_hat_dm = np.concatenate(dm_estimates, axis=0)
        nmse["dm"] = per_sample_nmse(channels, h_hat_dm)
    return EvaluationResult(nmse=nmse, y_spatial=y_spatial_np)


def nmse_db(values: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(values, np.finfo(np.float64).eps))


def make_bin_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot create bins from empty/non-finite values.")
    low, high = float(np.min(finite)), float(np.max(finite))
    if low == high:
        pad = max(abs(low) * 1e-6, 1e-6)
        return np.array([low - pad, high + pad])
    return np.linspace(low, high, num_bins + 1)


def bin_mask(values: np.ndarray, edges: np.ndarray, index: int) -> np.ndarray:
    if index == len(edges) - 2:
        return (values >= edges[index]) & (values <= edges[index + 1])
    return (values >= edges[index]) & (values < edges[index + 1])


def binned_rows(
    *,
    metric_name: str,
    values: np.ndarray,
    edges: np.ndarray,
    nmse_by_method: dict[str, np.ndarray],
    min_samples: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bin_index in range(len(edges) - 1):
        mask = bin_mask(values, edges, bin_index)
        center = 0.5 * (edges[bin_index] + edges[bin_index + 1])
        label = f"{edges[bin_index]:.3g}-{edges[bin_index + 1]:.3g}"
        n_samples = int(np.sum(mask))
        valid = n_samples >= min_samples
        for method, method_nmse in nmse_by_method.items():
            selected = method_nmse[mask]
            if valid:
                mean_nmse = float(np.mean(selected))
                std_nmse = float(np.std(selected, ddof=1)) if n_samples > 1 else 0.0
                sem = std_nmse / math.sqrt(n_samples) if n_samples > 1 else 0.0
                ci_low = max(mean_nmse - 1.96 * sem, np.finfo(np.float64).eps)
                ci_high = mean_nmse + 1.96 * sem
                row = {
                    "metric": metric_name,
                    "bin_index": bin_index,
                    "bin_low": edges[bin_index],
                    "bin_high": edges[bin_index + 1],
                    "bin_center": center,
                    "bin_label": label,
                    "method": method,
                    "n_samples": n_samples,
                    "valid": True,
                    "mean_nmse": mean_nmse,
                    "mean_nmse_db": nmse_db(mean_nmse),
                    "std_nmse": std_nmse,
                    "ci_low_db": nmse_db(ci_low),
                    "ci_high_db": nmse_db(ci_high),
                }
            else:
                row = {
                    "metric": metric_name,
                    "bin_index": bin_index,
                    "bin_low": edges[bin_index],
                    "bin_high": edges[bin_index + 1],
                    "bin_center": center,
                    "bin_label": label,
                    "method": method,
                    "n_samples": n_samples,
                    "valid": False,
                    "mean_nmse": "",
                    "mean_nmse_db": "",
                    "std_nmse": "",
                    "ci_low_db": "",
                    "ci_high_db": "",
                }
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_main_1x2(
    *,
    path: Path,
    doppler_rows: list[dict[str, object]],
    elevation_rows: list[dict[str, object]],
    methods: list[str],
    snr_db: float,
    run_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    panels = [
        (axes[0], doppler_rows, "Doppler", "Max Doppler (kHz)", 1e-3),
        (axes[1], elevation_rows, "Elevation", "Elevation (deg)", 1.0),
    ]
    for axis, rows, metric, xlabel, x_scale in panels:
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method and row["valid"]]
            if not method_rows:
                continue
            x = np.asarray([float(row["bin_center"]) * x_scale for row in method_rows])
            y = np.asarray([float(row["mean_nmse_db"]) for row in method_rows])
            ci_low = np.asarray([float(row["ci_low_db"]) for row in method_rows])
            ci_high = np.asarray([float(row["ci_high_db"]) for row in method_rows])
            yerr = np.vstack([y - ci_low, ci_high - y])
            axis.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=3, label=METHOD_LABELS[method])
        axis.set_title(f"NMSE vs {metric}")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("NMSE (dB)")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"{run_label}, SNR={snr_db:g} dB")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def heatmap_matrix(
    *,
    doppler: np.ndarray,
    elevation: np.ndarray,
    doppler_edges: np.ndarray,
    elevation_edges: np.ndarray,
    fixed_eval: EvaluationResult,
    heatmap_value: str,
    min_samples: int,
) -> np.ndarray:
    matrix = np.full((len(elevation_edges) - 1, len(doppler_edges) - 1), np.nan)
    for elev_index in range(len(elevation_edges) - 1):
        elev_mask = bin_mask(elevation, elevation_edges, elev_index)
        for dopp_index in range(len(doppler_edges) - 1):
            mask = elev_mask & bin_mask(doppler, doppler_edges, dopp_index)
            if int(np.sum(mask)) < min_samples:
                continue
            if heatmap_value == "dm":
                matrix[elev_index, dopp_index] = nmse_db(np.mean(fixed_eval.nmse["dm"][mask]))
            else:
                dm_db = nmse_db(np.mean(fixed_eval.nmse["dm"][mask]))
                lmmse_db = nmse_db(np.mean(fixed_eval.nmse["lmmse"][mask]))
                matrix[elev_index, dopp_index] = dm_db - lmmse_db
    return matrix


def plot_heatmap(
    *,
    path: Path,
    matrix: np.ndarray,
    doppler_edges: np.ndarray,
    elevation_edges: np.ndarray,
    heatmap_value: str,
    snr_db: float,
    run_label: str,
) -> None:
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#eeeeee")
    fig, axis = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, aspect="auto", origin="lower", cmap=cmap)
    axis.set_title(f"Doppler-Elevation robustness, SNR={snr_db:g} dB")
    axis.set_xlabel("Max Doppler bin (kHz)")
    axis.set_ylabel("Elevation bin (deg)")
    x_labels = [f"{0.5 * (doppler_edges[i] + doppler_edges[i + 1]) / 1e3:.2g}" for i in range(len(doppler_edges) - 1)]
    y_labels = [f"{0.5 * (elevation_edges[i] + elevation_edges[i + 1]):.2g}" for i in range(len(elevation_edges) - 1)]
    axis.set_xticks(np.arange(len(x_labels)), x_labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(y_labels)), y_labels)
    label = "DM NMSE (dB)" if heatmap_value == "dm" else "DM - LMMSE NMSE (dB)"
    fig.colorbar(image, ax=axis, label=label)
    fig.suptitle(run_label, y=1.03)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_stress_groups(
    doppler: np.ndarray,
    elevation: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    high_doppler_mask = doppler >= np.nanpercentile(doppler, 80)
    low_elevation_mask = elevation <= np.nanpercentile(elevation, 20)
    normal_mask = (
        (doppler >= np.nanpercentile(doppler, 40))
        & (doppler <= np.nanpercentile(doppler, 60))
        & (elevation >= np.nanpercentile(elevation, 40))
        & (elevation <= np.nanpercentile(elevation, 60))
    )
    masks = {
        "normal": normal_mask,
        "low_elevation": low_elevation_mask,
        "high_doppler": high_doppler_mask,
    }
    groups: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        if indices.size > max_samples:
            indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
        groups[name] = indices
    return groups


def evaluate_stress_curves(
    *,
    run: LoadedRun,
    groups: dict[str, np.ndarray],
    stress_snrs: list[float],
    methods: list[str],
    diffusion_model: DiffusionModel,
    angular_variance: np.ndarray,
    fft_pre: bool,
    mode: str,
    device: str,
    batch_size: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    curves: dict[str, dict[str, list[float]]] = {
        group: {method: [] for method in methods}
        for group in groups
    }
    for snr_index, snr_db in enumerate(stress_snrs):
        for group, indices in groups.items():
            result = evaluate_methods(
                channels=run.test_channels[indices],
                snr_db=snr_db,
                seed=seed + 1009 * snr_index,
                methods=methods,
                diffusion_model=diffusion_model,
                angular_variance=angular_variance,
                fft_pre=fft_pre,
                mode=mode,
                device=device,
                batch_size=batch_size,
            )
            for method in methods:
                curves[group][method].append(float(np.mean(result.nmse[method])))
    return curves


def plot_stress_curves(
    *,
    path: Path,
    curves: dict[str, dict[str, list[float]]],
    stress_snrs: list[float],
    methods: list[str],
    run_label: str,
) -> None:
    fig, axes = plt.subplots(1, len(curves), figsize=(5.4 * len(curves), 4.8), constrained_layout=True)
    if len(curves) == 1:
        axes = [axes]
    for axis, (group, group_curves) in zip(axes, curves.items()):
        for method in methods:
            axis.plot(stress_snrs, nmse_db(np.asarray(group_curves[method])), marker="o", linewidth=2, label=METHOD_LABELS[method])
        axis.set_title(group.replace("_", " ").title())
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("Mean NMSE (dB)")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Stress scenario NMSE-SNR curves: {run_label}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stress_boxplot(
    *,
    path: Path,
    fixed_eval: EvaluationResult,
    groups: dict[str, np.ndarray],
    methods: list[str],
    snr_db: float,
    run_label: str,
) -> None:
    fig, axis = plt.subplots(figsize=(max(8.5, 1.5 * len(groups) * len(methods)), 5.2), constrained_layout=True)
    data: list[np.ndarray] = []
    labels: list[str] = []
    for group, indices in groups.items():
        for method in methods:
            data.append(nmse_db(fixed_eval.nmse[method][indices]))
            labels.append(f"{group.replace('_', ' ')}\n{METHOD_LABELS[method]}")
    axis.boxplot(data, labels=labels, showmeans=True)
    axis.set_ylabel("Per-sample NMSE (dB)")
    axis.set_title(f"Stress sample NMSE distributions, SNR={snr_db:g} dB")
    axis.grid(True, axis="y", alpha=0.3)
    fig.suptitle(run_label, y=1.03)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def per_sample_rows(
    *,
    run_label: str,
    doppler: np.ndarray,
    elevation: np.ndarray,
    fixed_eval: EvaluationResult,
    methods: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(doppler.shape[0]):
        row: dict[str, object] = {
            "run": run_label,
            "sample_index": index,
            "max_doppler_hz": doppler[index],
            "elevation_deg": elevation[index],
        }
        for method in methods:
            value = float(fixed_eval.nmse[method][index])
            row[f"nmse_{method}"] = value
            row[f"nmse_{method}_db"] = nmse_db(value)
        rows.append(row)
    return rows


def output_prefix(output_dir: Path, run_label: str, snr_db: float, methods: list[str]) -> Path:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snr_label = f"{snr_db:g}".replace("-", "neg").replace(".", "p")
    methods_label = "-".join(methods)
    return output_dir / f"{timestamp}_{run_label}_snr{snr_label}dB_{methods_label}"


def main() -> None:
    args = parse_args()
    if args.num_bins < 1:
        raise ValueError("--num-bins must be at least 1.")
    if args.min_bin_samples < 1:
        raise ValueError("--min-bin-samples must be at least 1.")

    device = choose_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested {device}, but CUDA is not available.")

    model_dir = Path(args.model_dir).resolve() if args.model_dir else latest_model_dir(PROJECT_DIR / "results")
    run = load_run(model_dir)
    diffusion_model = load_diffusion_model(run.params, run.model_dir, device)
    data_dict = run.params["data_dict"]
    tester_dict = run.params.get("tester_dict", {})
    fft_pre = bool(tester_dict.get("fft_pre", run.params.get("trainer_dict", {}).get("fft_pre", False)))
    mode = tester_dict.get("mode") or data_dict.get("mode", "2D")
    batch_size = args.batch_size or int(tester_dict.get("batch_size", 512))
    methods = [method for method in METHOD_ORDER if method in set(args.methods)]
    if args.heatmap_value == "dm-minus-lmmse" and not {"dm", "lmmse"}.issubset(methods):
        raise ValueError("--heatmap-value dm-minus-lmmse requires --methods dm lmmse.")

    angular_variance = estimate_angular_variance(run.train_channels)
    fixed_eval = evaluate_methods(
        channels=run.test_channels,
        snr_db=args.snr_db,
        seed=args.seed,
        methods=methods,
        diffusion_model=diffusion_model,
        angular_variance=angular_variance,
        fft_pre=fft_pre,
        mode=mode,
        device=device,
        batch_size=batch_size,
    )

    doppler_edges = make_bin_edges(run.doppler_hz, args.num_bins)
    elevation_edges = make_bin_edges(run.elevation_deg, args.num_bins)
    doppler_rows = binned_rows(
        metric_name="doppler_hz",
        values=run.doppler_hz,
        edges=doppler_edges,
        nmse_by_method=fixed_eval.nmse,
        min_samples=args.min_bin_samples,
    )
    elevation_rows = binned_rows(
        metric_name="elevation_deg",
        values=run.elevation_deg,
        edges=elevation_edges,
        nmse_by_method=fixed_eval.nmse,
        min_samples=args.min_bin_samples,
    )
    summary_rows = doppler_rows + elevation_rows

    output_dir = Path(args.output_dir).resolve()
    prefix = output_prefix(output_dir, run.model_dir.name, args.snr_db, methods)
    per_sample_path = prefix.with_name(f"{prefix.name}_per_sample.csv")
    summary_path = prefix.with_name(f"{prefix.name}_binned_summary.csv")
    main_path = prefix.with_name(f"{prefix.name}_main_1x2.png")
    heatmap_path = prefix.with_name(f"{prefix.name}_doppler_elevation_heatmap.png")
    stress_path = prefix.with_name(f"{prefix.name}_stress_nmse_snr.png")
    boxplot_path = prefix.with_name(f"{prefix.name}_stress_boxplot.png")

    per_fields = ["run", "sample_index", "max_doppler_hz", "elevation_deg"]
    for method in methods:
        per_fields += [f"nmse_{method}", f"nmse_{method}_db"]
    write_csv(
        per_sample_path,
        per_sample_rows(
            run_label=run.model_dir.name,
            doppler=run.doppler_hz,
            elevation=run.elevation_deg,
            fixed_eval=fixed_eval,
            methods=methods,
        ),
        per_fields,
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "metric",
            "bin_index",
            "bin_low",
            "bin_high",
            "bin_center",
            "bin_label",
            "method",
            "n_samples",
            "valid",
            "mean_nmse",
            "mean_nmse_db",
            "std_nmse",
            "ci_low_db",
            "ci_high_db",
        ],
    )

    plot_main_1x2(
        path=main_path,
        doppler_rows=doppler_rows,
        elevation_rows=elevation_rows,
        methods=methods,
        snr_db=args.snr_db,
        run_label=run.model_dir.name,
    )
    heatmap = heatmap_matrix(
        doppler=run.doppler_hz,
        elevation=run.elevation_deg,
        doppler_edges=doppler_edges,
        elevation_edges=elevation_edges,
        fixed_eval=fixed_eval,
        heatmap_value=args.heatmap_value,
        min_samples=args.min_bin_samples,
    )
    plot_heatmap(
        path=heatmap_path,
        matrix=heatmap,
        doppler_edges=doppler_edges,
        elevation_edges=elevation_edges,
        heatmap_value=args.heatmap_value,
        snr_db=args.snr_db,
        run_label=run.model_dir.name,
    )

    groups = select_stress_groups(
        run.doppler_hz,
        run.elevation_deg,
        max_samples=args.max_stress_samples,
        seed=args.seed,
    )
    stress_curves = evaluate_stress_curves(
        run=run,
        groups=groups,
        stress_snrs=list(args.stress_snrs),
        methods=methods,
        diffusion_model=diffusion_model,
        angular_variance=angular_variance,
        fft_pre=fft_pre,
        mode=mode,
        device=device,
        batch_size=batch_size,
        seed=args.seed,
    )
    plot_stress_curves(
        path=stress_path,
        curves=stress_curves,
        stress_snrs=list(args.stress_snrs),
        methods=methods,
        run_label=run.model_dir.name,
    )
    plot_stress_boxplot(
        path=boxplot_path,
        fixed_eval=fixed_eval,
        groups=groups,
        methods=methods,
        snr_db=args.snr_db,
        run_label=run.model_dir.name,
    )

    print(f"Saved main figure: {main_path}")
    print(f"Saved Doppler-elevation heatmap: {heatmap_path}")
    print(f"Saved stress NMSE-SNR figure: {stress_path}")
    print(f"Saved stress boxplot: {boxplot_path}")
    print(f"Saved per-sample CSV: {per_sample_path}")
    print(f"Saved binned summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
