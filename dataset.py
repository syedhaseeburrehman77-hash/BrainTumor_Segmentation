from __future__ import annotations

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RAY_ENABLE_WINDOWS_JOB_OBJECT"] = "0"
sys.modules.setdefault("tensorflow", None)

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
    Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    MapLabelValued, NormalizeIntensityd, Orientationd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, Spacingd,
)

PATCH_SIZE = (96, 96, 96)


def _case_files(subject_dir: Path, require_label: bool = True) -> dict:
    files = list(subject_dir.rglob("*.nii")) + list(subject_dir.rglob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {subject_dir}")

    def find(modality: str) -> Path:
        matches = [p for p in files if modality in p.name.lower() and "seg" not in p.name.lower()]
        if not matches:
            raise FileNotFoundError(f"Missing {modality} image in {subject_dir}")
        return sorted(matches, key=lambda p: len(p.name))[0]

    # Test t1ce before t1, otherwise T1ce can be incorrectly selected as T1.
    images = [str(find(name)) for name in ("t1ce", "t1", "t2", "flair")]
    record = {"image": images, "subject_id": subject_dir.name}
    labels = [p for p in files if "seg" in p.name.lower()]
    if require_label:
        if len(labels) != 1:
            raise FileNotFoundError(f"Expected exactly one segmentation in {subject_dir}, found {len(labels)}")
        record["label"] = str(labels[0])
    return record


def read_partitioning(data_root: str | Path, partition_csv: str | Path) -> list[tuple[str, list[dict]]]:
    """Return sorted (Partition_ID, labelled subject-records) pairs from FeTS CSV."""
    root = Path(data_root)
    table = pd.read_csv(partition_csv)
    required = {"Partition_ID", "Subject_ID"}
    if not required.issubset(table.columns):
        raise ValueError(f"{partition_csv} must contain {sorted(required)}; got {list(table.columns)}")
    groups = []
    for partition_id, frame in table.groupby("Partition_ID", sort=True):
        records = [_case_files(root / str(subject_id)) for subject_id in frame["Subject_ID"]]
        groups.append((str(partition_id), records))
    if not groups:
        raise ValueError("Partition CSV contains no clients")
    return groups


def client_records(data_root: str | Path, partition_csv: str | Path, partition_index: int) -> list[dict]:
    groups = read_partitioning(data_root, partition_csv)
    if not 0 <= partition_index < len(groups):
        raise IndexError(f"partition-id {partition_index} is invalid; dataset has {len(groups)} partitions")
    return groups[partition_index][1]


def _train_transforms():
    return Compose([
        LoadImaged(keys=("image", "label")), EnsureChannelFirstd(keys=("image", "label")),
        Orientationd(keys=("image", "label"), axcodes="RAS"),
        Spacingd(keys=("image", "label"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        MapLabelValued(keys="label", orig_labels=[4], target_labels=[3]),
        CropForegroundd(keys=("image", "label"), source_key="image"),
        RandCropByPosNegLabeld(keys=("image", "label"), label_key="label", spatial_size=PATCH_SIZE,
                               pos=1, neg=1, num_samples=2, image_key="image", image_threshold=0),
        RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
        RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),
        RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3),
        EnsureTyped(keys=("image", "label")),
    ])


def _val_transforms():
    return Compose([
        LoadImaged(keys=("image", "label")), EnsureChannelFirstd(keys=("image", "label")),
        Orientationd(keys=("image", "label"), axcodes="RAS"),
        Spacingd(keys=("image", "label"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        MapLabelValued(keys="label", orig_labels=[4], target_labels=[3]),
        CropForegroundd(keys=("image", "label"), source_key="image"),
        EnsureTyped(keys=("image", "label")),
    ])


def split_records(records: list[dict], validation_fraction: float = 0.15, seed: int = 42):
    """Deterministic local hold-out split; never mixes patients between institutions."""
    if len(records) < 2:
        return records, records
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n_val = max(1, int(round(len(records) * validation_fraction)))
    val_indices = set(order[:n_val].tolist())
    return [r for i, r in enumerate(records) if i not in val_indices], [r for i, r in enumerate(records) if i in val_indices]


def make_loaders(
    records: list[dict],
    batch_size: int,
    cache_rate: float = 0.0,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    train_records, val_records = split_records(records, seed=seed)
    if cache_rate:
        train_ds = CacheDataset(train_records, transform=_train_transforms(), cache_rate=cache_rate)
        val_ds = CacheDataset(val_records, transform=_val_transforms(), cache_rate=cache_rate)
    else:
        train_ds = Dataset(train_records, transform=_train_transforms())
        val_ds = Dataset(val_records, transform=_val_transforms())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    )


def fets_region_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute Dice/HD95 for ET, TC, WT from contiguous labels 0..3."""
    prediction = torch.argmax(logits, dim=1)
    target = labels[:, 0] if labels.ndim == 5 else labels
    pred_regions = torch.stack((prediction == 3, (prediction == 1) | (prediction == 3), prediction > 0), dim=1).float()
    target_regions = torch.stack((target == 3, (target == 1) | (target == 3), target > 0), dim=1).float()
    dice = DiceMetric(include_background=True, reduction="mean_batch")(pred_regions, target_regions).detach().cpu().numpy()
    hd95 = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")(pred_regions, target_regions).detach().cpu().numpy()
    dice = np.nan_to_num(dice, nan=0.0, posinf=0.0, neginf=0.0).flatten()
    hd95 = np.nan_to_num(hd95, nan=0.0, posinf=0.0, neginf=0.0).flatten()
    return {
        "dice_et": float(dice[0]),
        "dice_tc": float(dice[1]),
        "dice_wt": float(dice[2]),
        "hd95_et": float(hd95[0]),
        "hd95_tc": float(hd95[1]),
        "hd95_wt": float(hd95[2]),
    }
