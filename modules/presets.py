"""Shared dataset presets used by training and plotting entrypoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    channel_type: str
    train_files: tuple[str, ...]
    val_files: tuple[str, ...]
    test_files: tuple[str, ...] | None
    run_pattern: str


P006_TRAIN_FILE = "dataset/p006/LEO_Rural_LOS_h1000km_el30-90_path10_seed1111.mat"
P006_VAL_FILE = "dataset/p006/LEO_Rural_LOS_h1000km_el30-90_path10_seed2222.mat"
P006_RUN_PATTERN = "*leo_Rural_LOS_h1000km_el30-90_path10_tr1111_valtest2222"

P025_TRAIN_FILE = "dataset/p025/LEO_Urban_Mixed_h1000km_el15-90_path10_seed1111.mat"
P025_VAL_FILE = "dataset/p025/LEO_Urban_Mixed_h1000km_el15-90_path10_seed2222.mat"
P025_RUN_PATTERN = "*leo_Urban_Mixed_h1000km_el15-90_path10_tr1111_valtest2222"


DATASET_PRESETS = {
    "p006": DatasetPreset(
        name="p006",
        channel_type="leo",
        train_files=(P006_TRAIN_FILE,),
        val_files=(P006_VAL_FILE,),
        test_files=None,
        run_pattern=P006_RUN_PATTERN,
    ),
    "p025": DatasetPreset(
        name="p025",
        channel_type="leo",
        train_files=(P025_TRAIN_FILE,),
        val_files=(P025_VAL_FILE,),
        test_files=None,
        run_pattern=P025_RUN_PATTERN,
    ),
}


def get_dataset_preset(name: str) -> DatasetPreset:
    try:
        return DATASET_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASET_PRESETS))
        raise ValueError(f"Unknown dataset preset {name!r}. Available presets: {choices}") from exc
