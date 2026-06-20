#!/usr/bin/env python3
"""Plot clean, noisy, and truncated-DM-denoised channel heatmaps."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot original, SNR-corrupted, and truncated-DM-denoised channel heatmaps."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Result directory containing sim_params.json and train_models/. Defaults to latest results/* model dir.",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--snr-db", type=float, default=-5.0, help="AWGN SNR in dB.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index inside the model test file.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed used for the noisy sample.")
    parser.add_argument(
        "--domain",
        choices=("angular", "spatial"),
        default="angular",
        help="Plot angle-domain spectra or spatial-domain channel matrices.",
    )
    parser.add_argument(
        "--plot-form",
        choices=("2d", "3d", "both"),
        default="both",
        help="Save 2D heatmaps, 3D surfaces, or both.",
    )
    parser.add_argument(
        "--no-fftshift",
        action="store_true",
        help="Keep raw FFT bin order instead of centering the angle-domain spectrum.",
    )
    parser.add_argument(
        "--plot-value",
        choices=("magnitude", "magnitude-db", "power-db"),
        default="magnitude-db",
        help="Quantity shown in the heatmaps.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to results/channel_heatmaps/<timestamp>_snr...png.",
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
    import json

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


def resolve_data_files(params: dict) -> list[str]:
    data_dict = params["data_dict"]
    files = data_dict.get("test_files") or data_dict.get("val_files")
    if not files:
        raise ValueError("sim_params.json does not contain test_files or val_files for plotting.")
    return [str((PROJECT_DIR / file_path).resolve()) if not Path(file_path).is_absolute() else file_path for file_path in files]


def complex_channels_to_tensor(channels: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(channels[:, None, :]))
    return cmplx2real(tensor, dim=1, new_dim=False).float()


def tensor_to_complex_matrix(tensor: torch.Tensor) -> np.ndarray:
    complex_tensor = real2cmplx(tensor.detach().cpu(), dim=1)
    return complex_tensor.numpy()[0]


def apply_awgn(x: torch.Tensor, snr_linear: float, noise_multiplier: float) -> torch.Tensor:
    return x + noise_multiplier / np.sqrt(snr_linear) * torch.randn_like(x)


def heatmap_values(channel: np.ndarray, plot_value: str) -> np.ndarray:
    magnitude = np.abs(channel)
    eps = np.finfo(np.float32).eps
    if plot_value == "magnitude":
        return magnitude
    if plot_value == "magnitude-db":
        return 20.0 * np.log10(magnitude + eps)
    if plot_value == "power-db":
        return 10.0 * np.log10(np.square(magnitude) + eps)
    raise ValueError(plot_value)


def value_label(plot_value: str) -> str:
    return {
        "magnitude": "|H|",
        "magnitude-db": "20log10(|H|)",
        "power-db": "10log10(|H|^2)",
    }[plot_value]


def color_limits(images: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([image.reshape(-1) for image in images])
    vmin, vmax = np.nanpercentile(values, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    return float(vmin), float(vmax)


def nmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sum(np.abs(estimate - reference) ** 2) / np.sum(np.abs(reference) ** 2))


def axis_labels(domain: str) -> tuple[str, str]:
    if domain == "angular":
        return "TX angle bin", "RX angle bin"
    return "TX antenna index", "RX antenna index"


def prepare_plot_channels(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    *,
    domain: str,
    fftshift: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if domain == "angular" and fftshift:
        return tuple(np.fft.fftshift(channel, axes=(-2, -1)) for channel in (clean, noisy, denoised))
    return clean, noisy, denoised


def panel_titles(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    *,
    snr_db: float,
    domain: str,
) -> list[str]:
    domain_label = "Angle domain" if domain == "angular" else "Spatial domain"
    return [
        f"Original channel\n{domain_label}",
        f"Noisy channel, SNR={snr_db:g} dB\nNMSE={nmse(clean, noisy):.3e}",
        f"Truncated DM denoised\nNMSE={nmse(clean, denoised):.3e}",
    ]


def plot_2d_heatmaps(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    *,
    output_path: Path,
    plot_value: str,
    snr_db: float,
    sample_index: int,
    model_dir: Path,
    domain: str,
    fftshift: bool,
) -> None:
    clean, noisy, denoised = prepare_plot_channels(
        clean, noisy, denoised, domain=domain, fftshift=fftshift
    )
    heatmaps = [
        heatmap_values(clean, plot_value),
        heatmap_values(noisy, plot_value),
        heatmap_values(denoised, plot_value),
    ]
    vmin, vmax = color_limits(heatmaps)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    titles = panel_titles(clean, noisy, denoised, snr_db=snr_db, domain=domain)
    xlabel, ylabel = axis_labels(domain)
    last_image = None
    for axis, values, title in zip(axes, heatmaps, titles):
        last_image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

    fig.colorbar(last_image, ax=axes, shrink=0.86, label=value_label(plot_value))
    fig.suptitle(f"{model_dir.name}, sample {sample_index}", y=1.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_3d_surfaces(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    *,
    output_path: Path,
    plot_value: str,
    snr_db: float,
    sample_index: int,
    model_dir: Path,
    domain: str,
    fftshift: bool,
) -> None:
    clean, noisy, denoised = prepare_plot_channels(
        clean, noisy, denoised, domain=domain, fftshift=fftshift
    )
    surfaces = [
        heatmap_values(clean, plot_value),
        heatmap_values(noisy, plot_value),
        heatmap_values(denoised, plot_value),
    ]
    zmin, zmax = color_limits(surfaces)
    rx_bins = np.arange(surfaces[0].shape[0])
    tx_bins = np.arange(surfaces[0].shape[1])
    tx_grid, rx_grid = np.meshgrid(tx_bins, rx_bins)

    fig = plt.figure(figsize=(19, 6.5), constrained_layout=True)
    titles = panel_titles(clean, noisy, denoised, snr_db=snr_db, domain=domain)
    xlabel, ylabel = axis_labels(domain)
    last_surface = None
    for index, (values, title) in enumerate(zip(surfaces, titles), start=1):
        axis = fig.add_subplot(1, 3, index, projection="3d")
        last_surface = axis.plot_surface(
            tx_grid,
            rx_grid,
            values,
            cmap="viridis",
            vmin=zmin,
            vmax=zmax,
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel, labelpad=8)
        axis.set_ylabel(ylabel, labelpad=8)
        axis.set_zlabel(value_label(plot_value), labelpad=8)
        axis.set_zlim(zmin, zmax)
        axis.view_init(elev=28, azim=-135)

    fig.colorbar(last_surface, ax=fig.axes, shrink=0.62, pad=0.04, label=value_label(plot_value))
    fig.suptitle(f"{model_dir.name}, sample {sample_index}", y=1.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def default_output_base(snr_db: float, sample_index: int, plot_value: str, domain: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snr_label = f"{snr_db:g}".replace("-", "neg").replace(".", "p")
    return (
        PROJECT_DIR
        / "results"
        / "channel_heatmaps"
        / f"{timestamp}_sample{sample_index}_snr{snr_label}dB_{plot_value}_{domain}"
    )


def output_paths(output: str | None, *, snr_db: float, sample_index: int, plot_value: str, domain: str) -> dict[str, Path]:
    if output:
        base_path = Path(output).resolve()
        stem_path = base_path.with_suffix("")
    else:
        stem_path = default_output_base(snr_db, sample_index, plot_value, domain)
    return {
        "2d": stem_path.with_name(f"{stem_path.name}_2d").with_suffix(".png"),
        "3d": stem_path.with_name(f"{stem_path.name}_3d").with_suffix(".png"),
    }


def select_plot_matrices(
    *,
    domain: str,
    fft_pre: bool,
    mode: str,
    x_spatial: torch.Tensor,
    y_spatial: torch.Tensor,
    x_est_spatial: torch.Tensor,
    x_model: torch.Tensor,
    y_model: torch.Tensor,
    x_est_model: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if domain == "spatial":
        tensors = (x_spatial, y_spatial, x_est_spatial)
    elif fft_pre:
        tensors = (x_model, y_model, x_est_model)
    else:
        tensors = tuple(
            ut.complex_1d_fft(tensor, ifft=False, mode=mode)
            for tensor in (x_spatial, y_spatial, x_est_spatial)
        )
    return tuple(tensor_to_complex_matrix(tensor) for tensor in tensors)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested {device}, but CUDA is not available.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dir = Path(args.model_dir).resolve() if args.model_dir else latest_model_dir(PROJECT_DIR / "results")
    params = load_params(model_dir)
    cnn_dict, diff_model_dict = tupleize_model_dicts(params, device)

    cnn = CNN(**cnn_dict)
    diffusion_model = DiffusionModel(cnn, **diff_model_dict)
    checkpoint = torch.load(latest_checkpoint(model_dir), map_location=device)
    diffusion_model.load_state_dict(checkpoint["model"])
    diffusion_model.eval()

    data_files = resolve_data_files(params)
    channels = Channels(files=data_files, strict_dtype=True).channels
    sample_index = args.sample_index
    if sample_index < 0:
        sample_index += channels.shape[0]
    if sample_index < 0 or sample_index >= channels.shape[0]:
        raise IndexError(f"sample-index {args.sample_index} is outside [0, {channels.shape[0] - 1}]")

    power_factor = params["data_dict"].get("power_normalization_factor")
    if power_factor:
        channels = channels / float(power_factor)

    x_spatial = complex_channels_to_tensor(channels[sample_index : sample_index + 1]).to(device)
    fft_pre = bool(params.get("tester_dict", {}).get("fft_pre", params.get("trainer_dict", {}).get("fft_pre", False)))
    mode = params.get("tester_dict", {}).get("mode") or params.get("data_dict", {}).get("mode", "2D")
    x_model = ut.complex_1d_fft(x_spatial, ifft=False, mode=mode) if fft_pre else x_spatial

    snr_linear = 10 ** (args.snr_db / 10.0)
    y_model = apply_awgn(x_model, snr_linear, diffusion_model.noise_multiplier)
    with torch.no_grad():
        x_est_model = diffusion_model.generate_estimate(
            y_model,
            snr_linear,
            add_random=False,
            return_all_timesteps=False,
        )

    y_spatial = ut.complex_1d_fft(y_model, ifft=True, mode=mode) if fft_pre else y_model
    x_est_spatial = ut.complex_1d_fft(x_est_model, ifft=True, mode=mode) if fft_pre else x_est_model

    clean, noisy, denoised = select_plot_matrices(
        domain=args.domain,
        fft_pre=fft_pre,
        mode=mode,
        x_spatial=x_spatial,
        y_spatial=y_spatial,
        x_est_spatial=x_est_spatial,
        x_model=x_model,
        y_model=y_model,
        x_est_model=x_est_model,
    )

    paths = output_paths(
        args.output,
        snr_db=args.snr_db,
        sample_index=sample_index,
        plot_value=args.plot_value,
        domain=args.domain,
    )
    fftshift = args.domain == "angular" and not args.no_fftshift
    if args.plot_form in ("2d", "both"):
        plot_2d_heatmaps(
            clean,
            noisy,
            denoised,
            output_path=paths["2d"],
            plot_value=args.plot_value,
            snr_db=args.snr_db,
            sample_index=sample_index,
            model_dir=model_dir,
            domain=args.domain,
            fftshift=fftshift,
        )
        print(f"Saved 2D figure: {paths['2d']}")
    if args.plot_form in ("3d", "both"):
        plot_3d_surfaces(
            clean,
            noisy,
            denoised,
            output_path=paths["3d"],
            plot_value=args.plot_value,
            snr_db=args.snr_db,
            sample_index=sample_index,
            model_dir=model_dir,
            domain=args.domain,
            fftshift=fftshift,
        )
        print(f"Saved 3D figure: {paths['3d']}")
    print(f"Noisy NMSE: {nmse(clean, noisy):.6e}")
    print(f"Denoised NMSE: {nmse(clean, denoised):.6e}")


if __name__ == "__main__":
    main()
