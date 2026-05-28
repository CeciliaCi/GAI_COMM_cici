"""Dataset loading utilities for LEO DeepMIMO channel files.

The loader keeps the training code on spatial-domain complex channels while
also exposing the angular-domain, two-channel real feature maps expected by the
diffusion model pipeline.
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_RX_ANTENNAS = 16
EXPECTED_TX_ANTENNAS = 144
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class LEOFileInfo:
    path: Path
    scenario: str | None = None
    los: str | None = None
    height_km: int | None = None
    elevation: tuple[float, float] | None = None
    num_paths: int | None = None
    seed: int | None = None


@dataclass
class LEOFileData:
    info: LEOFileInfo
    channels: np.ndarray
    dataset_params: dict[str, Any]
    sample_info: dict[str, Any]
    geometry_features: np.ndarray | None
    geometry_feature_names: list[str]


def parse_leo_filename(file_path: str | os.PathLike[str]) -> LEOFileInfo:
    """Parse both legacy and detailed LEO dataset names."""
    path = Path(file_path)
    stem = path.stem
    if not stem.startswith("LEO_"):
        return LEOFileInfo(path=path)

    parts = stem[len("LEO_"):].split("_")
    seed = None
    scenario_parts: list[str] = []
    los = None
    height_km = None
    elevation = None
    num_paths = None

    for token in parts:
        token_lower = token.lower()
        if token_lower.startswith("seed"):
            seed_value = token[4:]
            seed = int(seed_value) if seed_value.isdigit() else None
        elif token in {"LOS", "NLOS"}:
            los = token
        elif token_lower.startswith("h") and token_lower.endswith("km"):
            height_value = token[1:-2]
            height_km = int(height_value) if height_value.isdigit() else None
        elif token_lower.startswith("el") and "-" in token:
            low, high = token[2:].split("-", maxsplit=1)
            elevation = (float(low), float(high))
        elif token_lower.startswith("path"):
            path_value = token[4:]
            num_paths = int(path_value) if path_value.isdigit() else None
        else:
            scenario_parts.append(token)

    scenario = "_".join(scenario_parts) if scenario_parts else None
    return LEOFileInfo(
        path=path,
        scenario=scenario,
        los=los,
        height_km=height_km,
        elevation=elevation,
        num_paths=num_paths,
        seed=seed,
    )


def match_leo_files(
    data_dir: str | os.PathLike[str] = "dataset",
    *,
    pattern: str | None = None,
    scenario: str | None = None,
    los: str | None = None,
    height_km: int | None = None,
    elevation: str | tuple[float, float] | None = None,
    num_paths: int | None = None,
    seeds: list[int] | tuple[int, ...] | None = None,
) -> list[LEOFileInfo]:
    """Return LEO files matching the filename-level physical filters."""
    root = Path(data_dir)
    if pattern:
        candidates = [Path(p) for p in glob.glob(str(root / pattern))]
    else:
        candidates = sorted(root.glob("LEO_*.mat"))

    elevation_tuple = _parse_elevation(elevation)
    seed_set = set(seeds) if seeds is not None else None
    matches: list[LEOFileInfo] = []
    for candidate in candidates:
        info = parse_leo_filename(candidate)
        if scenario is not None and info.scenario is not None and info.scenario.lower() != scenario.lower():
            continue
        if los is not None and info.los is not None and info.los.lower() != los.lower():
            continue
        if height_km is not None and info.height_km is not None and info.height_km != height_km:
            continue
        if elevation_tuple is not None and info.elevation is not None and info.elevation != elevation_tuple:
            continue
        if num_paths is not None and info.num_paths is not None and info.num_paths != num_paths:
            continue
        if seed_set is not None and info.seed is not None and info.seed not in seed_set:
            continue
        matches.append(info)

    return sorted(matches, key=_leo_sort_key)


class Channels:
    """Aggregated LEO channels plus metadata and split helpers."""

    def __init__(
        self,
        data_dir: str | os.PathLike[str] = "dataset",
        *,
        files: list[str | os.PathLike[str]] | None = None,
        pattern: str | None = None,
        scenario: str | None = None,
        los: str | None = None,
        height_km: int | None = None,
        elevation: str | tuple[float, float] | None = None,
        num_paths: int | None = None,
        seeds: list[int] | tuple[int, ...] | None = None,
        expected_rx: int = EXPECTED_RX_ANTENNAS,
        expected_tx: int = EXPECTED_TX_ANTENNAS,
        strict_dtype: bool = True,
    ):
        if files is None:
            self.files = match_leo_files(
                data_dir,
                pattern=pattern,
                scenario=scenario,
                los=los,
                height_km=height_km,
                elevation=elevation,
                num_paths=num_paths,
                seeds=seeds,
            )
        else:
            self.files = [parse_leo_filename(file_path) for file_path in files]

        if not self.files:
            raise FileNotFoundError(f"No LEO .mat files matched in {data_dir!s}")

        records = [
            load_leo_mat_file(
                file_info,
                expected_rx=expected_rx,
                expected_tx=expected_tx,
                strict_dtype=strict_dtype,
            )
            for file_info in self.files
        ]
        self.records = [
            record for record in records
            if _info_matches_filters(
                record.info,
                scenario=scenario,
                los=los,
                height_km=height_km,
                elevation=elevation,
                num_paths=num_paths,
                seeds=seeds,
            )
        ]
        if not self.records:
            raise FileNotFoundError("No LEO .mat files matched after metadata filtering")

        self.files = [record.info for record in self.records]
        self.channels = np.concatenate([record.channels for record in self.records], axis=0)
        self.dataset_params = [record.dataset_params for record in self.records]
        self.sample_info = [record.sample_info for record in self.records]
        self.geometry_feature_names, self.geometry_features = _merge_geometry_features(self.records)

    @property
    def num_samples(self) -> int:
        return int(self.channels.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.channels.shape

    def split(
        self,
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
        *,
        split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        shuffle: bool = True,
        seed: int | None = 453451,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return split_channels(
            self.channels,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            split_ratios=split_ratios,
            shuffle=shuffle,
            seed=seed,
        )

    def angular(self) -> np.ndarray:
        return angular_domain_transform(self.channels)

    def feature_maps(self, *, angular: bool = True, dtype: np.dtype = np.float32) -> np.ndarray:
        return complex_to_two_channel(self.angular() if angular else self.channels, dtype=dtype)

    def roundtrip_error(self) -> float:
        angular = self.angular()
        restored = inverse_angular_domain_transform(angular)
        return float(np.linalg.norm(self.channels - restored) ** 2)

    def summary(self) -> dict[str, Any]:
        first_params = self.dataset_params[0] if self.dataset_params else {}
        return {
            "num_files": len(self.files),
            "num_samples": self.num_samples,
            "channels_shape": self.channels.shape,
            "channels_dtype": str(self.channels.dtype),
            "scenario": _metadata_value(first_params, "scenario"),
            "scenario_type": _metadata_value(first_params, "scenario_type"),
            "orbit_height_km": _metadata_value(first_params, "orbit_height_km"),
            "num_paths": _metadata_value(first_params, "quadriga_num_subpaths"),
            "geometry_feature_shape": None
            if self.geometry_features is None
            else self.geometry_features.shape,
        }


def load_leo_mat_file(
    file_info: LEOFileInfo | str | os.PathLike[str],
    *,
    expected_rx: int = EXPECTED_RX_ANTENNAS,
    expected_tx: int = EXPECTED_TX_ANTENNAS,
    strict_dtype: bool = True,
) -> LEOFileData:
    """Load one LEO DeepMIMO .mat file and normalize channels to [N, 16, 144]."""
    info = file_info if isinstance(file_info, LEOFileInfo) else parse_leo_filename(file_info)
    raw = _load_mat_variables(info.path)
    if "channels" not in raw:
        raise KeyError(f"{info.path} does not contain a top-level 'channels' variable")

    channels = normalize_channel_tensor(
        raw["channels"],
        expected_rx=expected_rx,
        expected_tx=expected_tx,
        strict_dtype=strict_dtype,
    )
    dataset_params = _ensure_dict(raw.get("dataset_params", {}))
    sample_info = _ensure_dict(raw.get("sample_info", {}))
    geometry_names, geometry_features = extract_geometry_features(sample_info, channels.shape[0])
    merged_info = _merge_filename_and_metadata(info, dataset_params, sample_info)
    return LEOFileData(
        info=merged_info,
        channels=channels,
        dataset_params=dataset_params,
        sample_info=sample_info,
        geometry_features=geometry_features,
        geometry_feature_names=geometry_names,
    )


def load_leo_data_splits(
    data_dir: str | os.PathLike[str] = "dataset",
    *,
    files: list[str | os.PathLike[str]] | None = None,
    pattern: str | None = None,
    scenario: str | None = None,
    los: str | None = None,
    height_km: int | None = None,
    elevation: str | tuple[float, float] | None = None,
    num_paths: int | None = None,
    seeds: list[int] | tuple[int, ...] | None = None,
    n_train: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    shuffle: bool = True,
    seed: int | None = 453451,
    return_channels: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, Channels]:
    channels = Channels(
        data_dir,
        files=files,
        pattern=pattern,
        scenario=scenario,
        los=los,
        height_km=height_km,
        elevation=elevation,
        num_paths=num_paths,
        seeds=seeds,
    )
    train, val, test = channels.split(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        split_ratios=split_ratios,
        shuffle=shuffle,
        seed=seed,
    )
    if return_channels:
        return train, val, test, channels
    return train, val, test


def normalize_channel_tensor(
    channels: Any,
    *,
    expected_rx: int = EXPECTED_RX_ANTENNAS,
    expected_tx: int = EXPECTED_TX_ANTENNAS,
    strict_dtype: bool = True,
) -> np.ndarray:
    """Convert MATLAB/HDF5 channel storage to [num_users, rx, tx] complex128."""
    channels = _as_complex_array(channels)
    channels = np.squeeze(channels)
    if channels.ndim != 3:
        raise ValueError(f"Expected a 3-D channel tensor, got shape {channels.shape}")

    channels = _move_channel_axes(channels, expected_rx=expected_rx, expected_tx=expected_tx)
    if channels.shape[1:] != (expected_rx, expected_tx):
        raise ValueError(
            f"Expected channel shape [num_users, {expected_rx}, {expected_tx}], got {channels.shape}"
        )
    if not np.iscomplexobj(channels):
        raise TypeError("LEO channels must be complex-valued")
    if strict_dtype and channels.dtype != np.complex128:
        raise TypeError(f"LEO channels must be complex double/complex128, got {channels.dtype}")
    return np.ascontiguousarray(channels.astype(np.complex128, copy=False))


def angular_domain_transform(channels: np.ndarray) -> np.ndarray:
    """Unitary 2-D DFT over RX/TX antenna axes."""
    return np.fft.fft2(channels, axes=(-2, -1), norm="ortho")


def inverse_angular_domain_transform(channels: np.ndarray) -> np.ndarray:
    """Inverse unitary 2-D DFT over RX/TX antenna axes."""
    return np.fft.ifft2(channels, axes=(-2, -1), norm="ortho")


def complex_to_two_channel(channels: np.ndarray, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Map complex [N, RX, TX] data to real [N, 2, RX, TX] feature maps."""
    if not np.iscomplexobj(channels):
        raise TypeError("Input must be complex-valued")
    return np.stack([channels.real, channels.imag], axis=1).astype(dtype, copy=False)


def split_channels(
    channels: np.ndarray,
    *,
    n_train: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    shuffle: bool = True,
    seed: int | None = 453451,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split aggregated channels either by counts or by train/val/test ratios."""
    num_samples = int(channels.shape[0])
    counts = _resolve_split_counts(num_samples, n_train, n_val, n_test, split_ratios)
    indices = np.arange(num_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train_resolved, n_val_resolved, n_test_resolved = counts
    train_idx = indices[:n_train_resolved]
    val_idx = indices[n_train_resolved:n_train_resolved + n_val_resolved]
    test_idx = indices[n_train_resolved + n_val_resolved:n_train_resolved + n_val_resolved + n_test_resolved]
    return channels[train_idx], channels[val_idx], channels[test_idx]


def extract_geometry_features(
    sample_info: dict[str, Any],
    num_samples: int,
    *,
    max_features: int = 64,
) -> tuple[list[str], np.ndarray | None]:
    """Extract per-sample numeric geometry fields for future conditioning."""
    names: list[str] = []
    columns: list[np.ndarray] = []
    _collect_numeric_features(sample_info, num_samples, "", names, columns, max_features)
    if not columns:
        return [], None
    features = np.stack(columns, axis=1).astype(np.float32, copy=False)
    return names, features


def _load_mat_variables(file_path: Path) -> dict[str, Any]:
    if _is_hdf5_mat(file_path):
        return _load_hdf5_mat(file_path)
    return _load_scipy_mat(file_path)


def _load_hdf5_mat(file_path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading MATLAB 7.3/HDF5 .mat files requires h5py. Install it with "
            "`conda install -n dmce -c conda-forge h5py` or "
            "`python -m pip install --only-binary=:all: h5py`."
        ) from exc

    with h5py.File(file_path, "r") as handle:
        return {
            key: _read_hdf5_node(handle[key], handle)
            for key in handle.keys()
            if key != "#refs#"
        }


def _load_scipy_mat(file_path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("Reading pre-v7.3 .mat files requires scipy.") from exc

    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    return {key: _mat_to_python(value) for key, value in data.items() if not key.startswith("__")}


def _read_hdf5_node(node: Any, h5_file: Any) -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Reading HDF5 .mat files requires h5py.") from exc

    if isinstance(node, h5py.Group):
        return {key: _read_hdf5_node(node[key], h5_file) for key in node.keys()}
    if not isinstance(node, h5py.Dataset):
        return node

    data = node[()]
    ref_type = h5py.check_dtype(ref=node.dtype)
    if ref_type is not None:
        refs = np.asarray(data).reshape(-1)
        values = [_read_hdf5_node(h5_file[ref], h5_file) for ref in refs if ref]
        if np.asarray(data).size == 1:
            return values[0] if values else None
        return np.asarray(values, dtype=object).reshape(np.asarray(data).shape)

    if getattr(data, "dtype", None) is not None and data.dtype.names:
        names = set(data.dtype.names)
        if {"real", "imag"}.issubset(names):
            return data["real"] + 1j * data["imag"]

    matlab_class = _decode_attr(node.attrs.get("MATLAB_class"))
    if matlab_class == "char":
        return _decode_matlab_char(data)
    if matlab_class == "logical":
        return np.asarray(data).astype(bool).squeeze()

    data = np.asarray(data)
    if data.shape == (1, 1):
        return data.item()
    return np.squeeze(data)


def _mat_to_python(value: Any) -> Any:
    if hasattr(value, "_fieldnames"):
        return {field: _mat_to_python(getattr(value, field)) for field in value._fieldnames}
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return {
                name: _mat_to_python(value[name].squeeze())
                for name in value.dtype.names
            }
        if value.dtype == object:
            squeezed = value.squeeze()
            if squeezed.shape == ():
                return _mat_to_python(squeezed.item())
            return np.asarray([_mat_to_python(item) for item in squeezed.reshape(-1)], dtype=object).reshape(
                squeezed.shape
            )
    return value


def _as_complex_array(value: Any) -> np.ndarray:
    if isinstance(value, dict) and {"real", "imag"}.issubset(value):
        return np.asarray(value["real"]) + 1j * np.asarray(value["imag"])

    array = np.asarray(value)
    if array.dtype.names and {"real", "imag"}.issubset(set(array.dtype.names)):
        array = array["real"] + 1j * array["imag"]
    return array


def _move_channel_axes(channels: np.ndarray, *, expected_rx: int, expected_tx: int) -> np.ndarray:
    if channels.shape[-2:] == (expected_rx, expected_tx):
        return channels

    shape = channels.shape
    rx_axes = [axis for axis, size in enumerate(shape) if size == expected_rx]
    tx_axes = [axis for axis, size in enumerate(shape) if size == expected_tx]
    for rx_axis in rx_axes:
        for tx_axis in tx_axes:
            if rx_axis == tx_axis:
                continue
            sample_axes = [axis for axis in range(channels.ndim) if axis not in {rx_axis, tx_axis}]
            if len(sample_axes) == 1:
                return np.transpose(channels, (sample_axes[0], rx_axis, tx_axis))

    raise ValueError(
        f"Could not identify RX/TX axes {expected_rx}/{expected_tx} in channel shape {channels.shape}"
    )


def _resolve_split_counts(
    num_samples: int,
    n_train: int | None,
    n_val: int | None,
    n_test: int | None,
    split_ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    requested = [n_train, n_val, n_test]
    if all(count is None for count in requested):
        ratios = np.asarray(split_ratios, dtype=float)
        if ratios.shape != (3,) or np.any(ratios < 0) or np.sum(ratios) <= 0:
            raise ValueError("split_ratios must contain three non-negative values with a positive sum")
        ratios = ratios / np.sum(ratios)
        train = int(np.floor(num_samples * ratios[0]))
        val = int(np.floor(num_samples * ratios[1]))
        test = num_samples - train - val
        return train, val, test

    known_total = sum(count for count in requested if count is not None)
    if known_total > num_samples:
        raise ValueError(f"Requested {known_total} samples, but only {num_samples} are available")

    missing = [index for index, count in enumerate(requested) if count is None]
    if missing:
        remaining = num_samples - known_total
        ratios = np.asarray([split_ratios[index] for index in missing], dtype=float)
        if np.sum(ratios) <= 0:
            ratios = np.ones(len(missing), dtype=float)
        ratios = ratios / np.sum(ratios)
        filled = [int(np.floor(remaining * ratio)) for ratio in ratios]
        filled[-1] = remaining - sum(filled[:-1])
        for index, count in zip(missing, filled):
            requested[index] = count

    counts = tuple(int(count) for count in requested)
    if sum(counts) > num_samples:
        raise ValueError(f"Requested {sum(counts)} samples, but only {num_samples} are available")
    return counts


def _collect_numeric_features(
    value: Any,
    num_samples: int,
    prefix: str,
    names: list[str],
    columns: list[np.ndarray],
    max_features: int,
) -> None:
    if len(columns) >= max_features:
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _collect_numeric_features(nested, num_samples, name, names, columns, max_features)
            if len(columns) >= max_features:
                return
        return

    array = np.asarray(value)
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        return
    array = np.squeeze(array)
    if array.ndim == 0:
        return
    if array.shape[0] == num_samples:
        matrix = array.reshape(num_samples, -1)
    elif array.shape[-1] == num_samples:
        matrix = np.moveaxis(array, -1, 0).reshape(num_samples, -1)
    else:
        return

    for column_index in range(matrix.shape[1]):
        if len(columns) >= max_features:
            return
        column_name = prefix if matrix.shape[1] == 1 else f"{prefix}_{column_index}"
        names.append(column_name)
        columns.append(matrix[:, column_index].astype(np.float64, copy=False))


def _merge_geometry_features(records: list[LEOFileData]) -> tuple[list[str], np.ndarray | None]:
    records_with_features = [record for record in records if record.geometry_features is not None]
    if not records_with_features:
        return [], None

    names = records_with_features[0].geometry_feature_names
    compatible = all(record.geometry_feature_names == names for record in records_with_features)
    if not compatible:
        return [], None
    return names, np.concatenate([record.geometry_features for record in records_with_features], axis=0)


def _merge_filename_and_metadata(
    info: LEOFileInfo,
    dataset_params: dict[str, Any],
    sample_info: dict[str, Any],
) -> LEOFileInfo:
    scenario_type = _metadata_value(dataset_params, "scenario_type")
    height = _metadata_value(dataset_params, "orbit_height_km")
    subpaths = _metadata_value(dataset_params, "quadriga_num_subpaths")
    seed = _metadata_value(dataset_params, "seed")
    los_flag = _metadata_value(sample_info, "los_flag")

    los = info.los
    if los is None and los_flag is not None:
        los_array = np.asarray(los_flag).astype(bool)
        los = "LOS" if np.all(los_array) else "NLOS" if not np.any(los_array) else "MIXED"

    return LEOFileInfo(
        path=info.path,
        scenario=info.scenario or _scalar_to_string(scenario_type),
        los=los,
        height_km=info.height_km or _optional_int(height),
        elevation=info.elevation,
        num_paths=info.num_paths or _optional_int(subpaths),
        seed=info.seed or _optional_int(seed),
    )


def _info_matches_filters(
    info: LEOFileInfo,
    *,
    scenario: str | None,
    los: str | None,
    height_km: int | None,
    elevation: str | tuple[float, float] | None,
    num_paths: int | None,
    seeds: list[int] | tuple[int, ...] | None,
) -> bool:
    elevation_tuple = _parse_elevation(elevation)
    if scenario is not None and (info.scenario or "").lower() != scenario.lower():
        return False
    if los is not None and (info.los or "").lower() != los.lower():
        return False
    if height_km is not None and info.height_km != height_km:
        return False
    if elevation_tuple is not None and info.elevation != elevation_tuple:
        return False
    if num_paths is not None and info.num_paths != num_paths:
        return False
    if seeds is not None and info.seed not in set(seeds):
        return False
    return True


def _metadata_value(mapping: dict[str, Any], key: str) -> Any:
    key_lower = key.lower()
    for actual_key, value in mapping.items():
        if str(actual_key).lower() == key_lower:
            return value
    for value in mapping.values():
        if isinstance(value, dict):
            found = _metadata_value(value, key)
            if found is not None:
                return found
    return None


def _ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_hdf5_mat(file_path: Path) -> bool:
    with open(file_path, "rb") as handle:
        head = handle.read(128)
    return b"HDF" in head or head.startswith(b"\x89HDF")


def _decode_attr(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tobytes"):
        return value.tobytes().decode("utf-8").strip("\x00")
    return str(value)


def _decode_matlab_char(data: Any) -> str:
    array = np.asarray(data).squeeze()
    if array.dtype.kind == "S":
        return b"".join(array.reshape(-1)).decode("utf-8")
    chars = [chr(int(code)) for code in array.reshape(-1, order="F") if int(code) != 0]
    return "".join(chars)


def _parse_elevation(elevation: str | tuple[float, float] | None) -> tuple[float, float] | None:
    if elevation is None:
        return None
    if isinstance(elevation, tuple):
        return (float(elevation[0]), float(elevation[1]))
    low, high = elevation.replace("el", "").split("-", maxsplit=1)
    return (float(low), float(high))


def _leo_sort_key(info: LEOFileInfo) -> tuple[Any, ...]:
    elevation = info.elevation or (-1.0, -1.0)
    return (
        str(info.scenario or ""),
        str(info.los or ""),
        info.height_km if info.height_km is not None else -1,
        elevation[0],
        elevation[1],
        info.num_paths if info.num_paths is not None else -1,
        info.seed if info.seed is not None else -1,
        str(info.path),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    array = np.asarray(value).squeeze()
    if array.shape != ():
        return None
    try:
        return int(array.item())
    except (TypeError, ValueError):
        return None


def _scalar_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    array = np.asarray(value).squeeze()
    if array.shape == ():
        return str(array.item())
    return None


def _format_shape(shape: tuple[int, ...]) -> str:
    return "[" + " ".join(str(dim) for dim in shape) + "]"


def _print_summary(channels: Channels, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> None:
    summary = channels.summary()
    print(f"matched files: {summary['num_files']}")
    for record in channels.records:
        print(
            "file: "
            f"{record.info.path.name}, "
            f"scenario={record.info.scenario}, "
            f"los={record.info.los}, "
            f"height_km={record.info.height_km}, "
            f"paths={record.info.num_paths}, "
            f"seed={record.info.seed}"
        )
    print(f"channels size: {_format_shape(channels.channels.shape)}")
    print(f"channels dtype: {channels.channels.dtype}")
    print(f"angular feature size: {_format_shape(channels.feature_maps().shape)}")
    print(f"geometry feature size: {summary['geometry_feature_shape']}")
    print(f"roundtrip frobenius error: {channels.roundtrip_error():.6e}")
    print(
        "split sizes: "
        f"train={_format_shape(train.shape)}, "
        f"val={_format_shape(val.shape)}, "
        f"test={_format_shape(test.shape)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and split LEO DeepMIMO channel datasets.")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--los", default=None)
    parser.add_argument("--height-km", type=int, default=None)
    parser.add_argument("--elevation", default=None, help="Elevation interval such as 30-90")
    parser.add_argument("--paths", type=int, default=None, dest="num_paths")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--split-ratios", nargs=3, type=float, default=DEFAULT_SPLIT_RATIOS)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=453451)
    args = parser.parse_args()

    train, val, test, channels = load_leo_data_splits(
        data_dir=args.data_dir,
        pattern=args.pattern,
        scenario=args.scenario,
        los=args.los,
        height_km=args.height_km,
        elevation=args.elevation,
        num_paths=args.num_paths,
        seeds=args.seeds,
        n_train=args.train_samples,
        n_val=args.val_samples,
        n_test=args.test_samples,
        split_ratios=tuple(args.split_ratios),
        shuffle=not args.no_shuffle,
        seed=args.seed,
        return_channels=True,
    )
    _print_summary(channels, train, val, test)


if __name__ == "__main__":
    main()
