#!/usr/bin/env python3
"""Plot one Doppler frequency-shift vs Magnitude curve for off-grid Doppler."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import sys
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
from modules.presets import DATASET_PRESETS, get_dataset_preset
from modules import utils as ut


METHOD_LABELS = {
    "dm": "DM estimate",
    "lmmse": "LMMSE estimate",
    "ls": "LS estimate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a single Doppler frequency shift vs Magnitude curve."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Result directory containing sim_params.json and train_models/. Defaults to latest results/* model dir.",
    )
    parser.add_argument(
        "--run-pattern",
        default=None,
        help="Glob pattern for selecting the latest matching run under results/ when --model-dir is omitted.",
    )
    parser.add_argument(
        "--data-preset",
        choices=sorted(DATASET_PRESETS),
        default=None,
        help="Use a predefined run pattern when --model-dir and --run-pattern are omitted.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Index in the test set.")
    parser.add_argument("--snr-db", type=float, default=0.0, help="AWGN SNR in dB.")
    parser.add_argument(
        "--offset",
        type=float,
        default=0.5,
        help="Fractional Doppler-bin offset applied along the last channel dimension.",
    )
    parser.add_argument("--seed", type=int, default=10, help="Random seed for AWGN generation.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--batch-size", type=int, default=None, help="DM evaluation batch size.")
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=("dm", "lmmse", "ls", "grid"),
        default=["dm"],
        help="Optional estimators to overlay. Ground truth and noisy observation are always plotted. 'grid' is ignored.",
    )
    parser.add_argument(
        "--delay-bin",
        type=int,
        default=None,
        help="Delay-like FFT bin used for the Doppler slice. Defaults to the dominant ground-truth bin.",
    )
    parser.add_argument(
        "--doppler-bin-hz",
        type=float,
        default=None,
        help="Doppler grid spacing in Hz. Defaults to subcarrier_spacing_hz / channel_width when available.",
    )
    parser.add_argument(
        "--hide-peak-lines",
        action="store_true",
        help="Do not draw faint vertical lines for estimator peak locations.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "results" / "offgrid_doppler_magnitude"),
        help="Directory for the single generated PNG.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output PNG path.",
    )
    parser.add_argument("--offset-min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--offset-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-offsets", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--offset-grid", nargs="+", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-bins", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--min-bin-samples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-samples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stress-snrs", nargs="+", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-stress-samples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--heatmap-value", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_arg


def latest_model_dir(results_root: Path, pattern: str = "*") -> Path:
    if Path(pattern).is_absolute():
        raw_candidates = [Path(path) for path in glob.glob(pattern)]
    else:
        raw_candidates = list(results_root.glob(pattern))

    candidates = [
        path
        for path in raw_candidates
        if path.is_dir() and (path / "sim_params.json").is_file() and (path / "train_models").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No trained DM result directory matching {pattern!r} found under {results_root}"
        )
    return sorted(candidates, key=lambda path: path.name)[-1]


def resolve_model_dir(args: argparse.Namespace) -> Path:
    if args.model_dir and args.run_pattern:
        raise ValueError("--model-dir and --run-pattern cannot be used together.")
    if args.model_dir:
        return Path(args.model_dir).resolve()

    pattern = args.run_pattern
    if pattern is None and args.data_preset is not None:
        pattern = get_dataset_preset(args.data_preset).run_pattern

    return latest_model_dir(PROJECT_DIR / "results", pattern or "*").resolve()


def load_params(model_dir: Path) -> dict:
    with (model_dir / "sim_params.json").open("r") as handle:
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


def load_channels_and_params(files: list[str]) -> tuple[np.ndarray, dict]:
    channels = Channels(files=files, strict_dtype=True)
    dataset_params = channels.records[0].dataset_params if channels.records else {}
    return channels.channels, dataset_params


def load_data(model_dir: Path) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    params = load_params(model_dir)
    data_dict = params["data_dict"]
    train_files = resolve_files(data_dict.get("train_files"))
    test_files = resolve_files(data_dict.get("test_files") or data_dict.get("val_files"))
    if not train_files or not test_files:
        raise ValueError("sim_params.json must contain train_files and test_files/val_files.")

    train_channels, train_dataset_params = load_channels_and_params(train_files)
    test_channels, test_dataset_params = load_channels_and_params(test_files)

    power_factor = data_dict.get("power_normalization_factor")
    if power_factor:
        train_channels = train_channels / float(power_factor)
        test_channels = test_channels / float(power_factor)

    return params, train_channels, test_channels, test_dataset_params or train_dataset_params


def load_diffusion_model(params: dict, model_dir: Path, device: str) -> DiffusionModel:
    cnn_dict, diff_model_dict = tupleize_model_dicts(params, device)
    cnn = CNN(**cnn_dict)
    diffusion_model = DiffusionModel(cnn, **diff_model_dict)
    checkpoint = torch.load(latest_checkpoint(model_dir), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    diffusion_model.load_state_dict(state_dict)
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


def scalar_param(params: dict, key: str) -> float | None:
    if key not in params:
        return None
    value = np.asarray(params[key]).squeeze()
    if value.size != 1:
        return None
    return float(value)


def infer_doppler_bin_hz(dataset_params: dict, n_doppler_bins: int, override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--doppler-bin-hz must be positive.")
        return float(override)

    for key in ("doppler_bin_hz", "residual_doppler_bin_hz", "doppler_resolution_hz"):
        value = scalar_param(dataset_params, key)
        if value is not None and value > 0:
            return value

    subcarrier_spacing_hz = scalar_param(dataset_params, "subcarrier_spacing_hz")
    if subcarrier_spacing_hz is not None and subcarrier_spacing_hz > 0:
        return subcarrier_spacing_hz / float(n_doppler_bins)

    sampling_interval_s = scalar_param(dataset_params, "sampling_interval_s")
    if sampling_interval_s is not None and sampling_interval_s > 0:
        return 1.0 / (float(n_doppler_bins) * sampling_interval_s)

    raise ValueError(
        "Cannot infer Doppler bin spacing. Provide --doppler-bin-hz or include "
        "subcarrier_spacing_hz/sampling_interval_s in dataset_params."
    )


def dft_bin_frequencies(n_bins: int, bin_hz: float) -> np.ndarray:
    return (np.arange(n_bins) - n_bins // 2) * bin_hz


def wrap_frequency(freq_hz: float, n_bins: int, bin_hz: float) -> float:
    span = n_bins * bin_hz
    low = -0.5 * span
    return float(((freq_hz - low) % span) + low)


def apply_fractional_doppler_offset(channel: np.ndarray, offset: float) -> np.ndarray:
    if offset < -0.5 or offset > 0.5:
        raise ValueError("--offset should stay inside [-0.5, 0.5].")
    n_doppler = channel.shape[-1]
    sample_axis = np.arange(n_doppler, dtype=float)
    phase = np.exp(1j * 2.0 * np.pi * offset * sample_axis / float(n_doppler))
    return channel * phase[None, :]


def spectrum(channel: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(channel, axes=(-2, -1), norm="ortho"), axes=(-2, -1))


def dominant_delay_and_doppler_bin(channel: np.ndarray, delay_bin: int | None) -> tuple[int, int]:
    spec = spectrum(channel)
    n_delay, n_doppler = spec.shape
    if delay_bin is not None:
        if delay_bin < 0 or delay_bin >= n_delay:
            raise ValueError(f"--delay-bin must be in [0, {n_delay - 1}].")
        return delay_bin, int(np.argmax(np.abs(spec[delay_bin])))
    flat_index = int(np.argmax(np.abs(spec)))
    return flat_index // n_doppler, flat_index % n_doppler


def estimate_angular_variance(train_channels: np.ndarray) -> np.ndarray:
    train_ang = np.fft.fft2(train_channels, axes=(-2, -1), norm="ortho")
    return np.mean(np.abs(train_ang) ** 2, axis=0)


def angular_diag_lmmse(y: np.ndarray, angular_variance: np.ndarray, sigma2: float) -> np.ndarray:
    y_ang = np.fft.fft2(y, axes=(-2, -1), norm="ortho")
    h_hat_ang = (angular_variance / (angular_variance + sigma2))[None, :, :] * y_ang
    return np.fft.ifft2(h_hat_ang, axes=(-2, -1), norm="ortho")


def estimate_channels(
    *,
    clean_channel: np.ndarray,
    train_channels: np.ndarray,
    params: dict,
    model_dir: Path,
    methods: list[str],
    snr_db: float,
    seed: int,
    device: str,
    batch_size: int | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    set_noise_seed(seed)
    snr_linear = 10 ** (snr_db / 10.0)
    tester_dict = params.get("tester_dict", {})
    data_dict = params.get("data_dict", {})
    fft_pre = bool(tester_dict.get("fft_pre", params.get("trainer_dict", {}).get("fft_pre", False)))
    mode = tester_dict.get("mode") or data_dict.get("mode", "2D")

    x_spatial = complex_channels_to_tensor(clean_channel[None]).to(device)
    x_model = ut.complex_1d_fft(x_spatial, ifft=False, mode=mode) if fft_pre else x_spatial

    diffusion_model = load_diffusion_model(params, model_dir, device) if "dm" in methods else None
    noise_multiplier = diffusion_model.noise_multiplier if diffusion_model is not None else 1.0 / math.sqrt(2.0)
    y_model = x_model + noise_multiplier / math.sqrt(snr_linear) * torch.randn_like(x_model)
    y_spatial = ut.complex_1d_fft(y_model, ifft=True, mode=mode) if fft_pre else y_model
    noisy = tensor_to_complex_numpy(y_spatial)

    estimates: dict[str, np.ndarray] = {}
    if "dm" in methods:
        effective_batch_size = batch_size or int(tester_dict.get("batch_size", 512))
        with torch.no_grad():
            x_est_model = diffusion_model.generate_estimate(
                y_model[:effective_batch_size],
                snr_linear,
                add_random=False,
                return_all_timesteps=False,
            )
            x_est_spatial = ut.complex_1d_fft(x_est_model, ifft=True, mode=mode) if fft_pre else x_est_model
        estimates["dm"] = tensor_to_complex_numpy(x_est_spatial)[0]

    if "lmmse" in methods:
        angular_variance = estimate_angular_variance(train_channels)
        sigma2 = 10 ** (-snr_db / 10.0)
        estimates["lmmse"] = angular_diag_lmmse(noisy, angular_variance, sigma2)[0]

    if "ls" in methods:
        estimates["ls"] = noisy[0]

    return noisy[0], estimates


def curve_for_delay(channel: np.ndarray, delay_bin: int) -> np.ndarray:
    return np.abs(spectrum(channel)[delay_bin])


def output_path(args: argparse.Namespace, model_dir: Path) -> Path:
    if args.output:
        return Path(args.output).resolve()
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snr_label = f"{args.snr_db:g}".replace("-", "neg").replace(".", "p")
    offset_label = f"{args.offset:g}".replace("-", "neg").replace(".", "p")
    name = (
        f"{timestamp}_{model_dir.name}_sample{args.sample_index}_"
        f"snr{snr_label}dB_offset{offset_label}_doppler_magnitude.png"
    )
    return Path(args.output_dir).resolve() / name


def plot_doppler_magnitude(
    *,
    path: Path,
    frequency_hz: np.ndarray,
    true_doppler_hz: float,
    delay_bin: int,
    curves: dict[str, np.ndarray],
    peak_lines: bool,
    title: str,
) -> None:
    x_khz = frequency_hz / 1e3
    fig, axis = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)

    styles = {
        "Ground truth": {"linewidth": 2.4, "color": "black"},
        "Noisy observation": {"linewidth": 1.8, "alpha": 0.72},
        "DM estimate": {"linewidth": 2.0},
        "LMMSE estimate": {"linewidth": 1.8},
        "LS estimate": {"linewidth": 1.6, "alpha": 0.7},
    }
    for label, values in curves.items():
        axis.plot(x_khz, values, label=label, **styles.get(label, {"linewidth": 1.8}))
        if peak_lines and label not in {"Ground truth", "Noisy observation"}:
            peak_hz = frequency_hz[int(np.argmax(values))]
            axis.axvline(peak_hz / 1e3, linestyle=":", linewidth=1.1, alpha=0.55)

    axis.axvline(true_doppler_hz / 1e3, color="black", linestyle="--", linewidth=1.3, label="True Doppler")
    axis.set_xlabel("Doppler frequency shift (kHz)")
    axis.set_ylabel("Magnitude |H(f_D)|")
    axis.set_title(f"{title}\nDelay bin {delay_bin}")
    axis.grid(True, alpha=0.3)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.offset_grid:
        args.offset = float(max(args.offset_grid, key=abs))
    device = choose_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested {device}, but CUDA is not available.")

    model_dir = resolve_model_dir(args)
    params, train_channels, test_channels, dataset_params = load_data(model_dir)

    if args.sample_index < 0 or args.sample_index >= test_channels.shape[0]:
        raise ValueError(f"--sample-index must be in [0, {test_channels.shape[0] - 1}].")

    base_channel = test_channels[args.sample_index]
    clean_channel = apply_fractional_doppler_offset(base_channel, args.offset)
    doppler_bin_hz = infer_doppler_bin_hz(dataset_params, clean_channel.shape[-1], args.doppler_bin_hz)
    frequency_hz = dft_bin_frequencies(clean_channel.shape[-1], doppler_bin_hz)

    delay_bin, base_doppler_bin = dominant_delay_and_doppler_bin(base_channel, args.delay_bin)
    true_doppler_hz = wrap_frequency(
        frequency_hz[base_doppler_bin] + args.offset * doppler_bin_hz,
        clean_channel.shape[-1],
        doppler_bin_hz,
    )

    methods = [method for method in dict.fromkeys(args.methods) if method != "grid"]
    noisy_channel, estimates = estimate_channels(
        clean_channel=clean_channel,
        train_channels=train_channels,
        params=params,
        model_dir=model_dir,
        methods=methods,
        snr_db=args.snr_db,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
    )

    curves = {
        "Ground truth": curve_for_delay(clean_channel, delay_bin),
        "Noisy observation": curve_for_delay(noisy_channel, delay_bin),
    }
    for method in methods:
        if method in estimates:
            curves[METHOD_LABELS[method]] = curve_for_delay(estimates[method], delay_bin)

    path = output_path(args, model_dir)
    plot_doppler_magnitude(
        path=path,
        frequency_hz=frequency_hz,
        true_doppler_hz=true_doppler_hz,
        delay_bin=delay_bin,
        curves=curves,
        peak_lines=not args.hide_peak_lines,
        title=f"{model_dir.name}, sample {args.sample_index}, SNR={args.snr_db:g} dB, offset={args.offset:g}",
    )
    print(f"Saved Doppler-Magnitude figure: {path}")


if __name__ == "__main__":
    main()
