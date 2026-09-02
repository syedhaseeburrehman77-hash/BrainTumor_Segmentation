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


def _global_test_split(
    records: list[dict],
    global_test_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Reserve deterministic unseen cases from one institution for final global testing."""
    if not 0.0 <= global_test_fraction < 1.0:
        raise ValueError("global_test_fraction must be in [0.0, 1.0)")
    if global_test_fraction == 0.0 or len(records) < 2:
        return records, []

    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(records) * global_test_fraction)))
    test_indices = set(rng.permutation(len(records))[:n_test].tolist())
    trainval = [record for index, record in enumerate(records) if index not in test_indices]
    test = [record for index, record in enumerate(records) if index in test_indices]
    return trainval, test


# Global-test module: clients receive only train/validation cases, never final-test cases.
def client_records(
    data_root: str | Path,
    partition_csv: str | Path,
    partition_index: int,
    global_test_fraction: float = 0.0,
    seed: int = 42,
) -> list[dict]:
    groups = read_partitioning(data_root, partition_csv)
    if not 0 <= partition_index < len(groups):
        raise IndexError(f"partition-id {partition_index} is invalid; dataset has {len(groups)} partitions")
    trainval, _ = _global_test_split(
        groups[partition_index][1],
        global_test_fraction=global_test_fraction,
        seed=seed + partition_index,
    )
    return trainval


# Global-test module: collect every reserved unseen case after federated training completes.
def global_test_records(
    data_root: str | Path,
    partition_csv: str | Path,
    global_test_fraction: float,
    seed: int = 42,
) -> list[dict]:
    groups = read_partitioning(data_root, partition_csv)
    records = []
    for partition_index, (_, institution_records) in enumerate(groups):
        _, test = _global_test_split(
            institution_records,
            global_test_fraction=global_test_fraction,
            seed=seed + partition_index,
        )
        records.extend(test)
    if not records:
        raise ValueError("Global test split contains no cases; increase global-test-fraction")
    return records


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


# Global-test module: fixed full-volume loader without augmentation or a local split.
def make_global_test_loader(
    records: list[dict],
    num_workers: int = 0,
    pin_memory: bool = False,
):
    dataset = Dataset(records, transform=_val_transforms())
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


MAX_BRATS_DISTANCE_MM = 373.13  # Diagonal distance penalty (240x240x155) for missed lesions


def fets_region_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute Dice, HD95, and foreground voxel counts for ET, TC, WT.
    
    Empty predictions for present lesions are penalized with MAX_BRATS_DISTANCE_MM
    rather than misleading 0.0 mm.
    """
    prediction = torch.argmax(logits, dim=1)
    target = labels[:, 0] if labels.ndim == 5 else labels

    pred_et = (prediction == 3).float()
    pred_tc = ((prediction == 1) | (prediction == 3)).float()
    pred_wt = (prediction > 0).float()

    target_et = (target == 3).float()
    target_tc = ((target == 1) | (target == 3)).float()
    target_wt = (target > 0).float()

    pred_regions = torch.stack((pred_et, pred_tc, pred_wt), dim=1)
    target_regions = torch.stack((target_et, target_tc, target_wt), dim=1)

    dice_metric = DiceMetric(include_background=True, reduction="none")
    dice_scores = dice_metric(pred_regions, target_regions).detach().cpu().numpy()

    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")
    hd95_scores = hd95_metric(pred_regions, target_regions).detach().cpu().numpy()

    results = {}
    region_names = ["et", "tc", "wt"]
    for i, name in enumerate(region_names):
        p_vox = float(pred_regions[:, i].sum().item())
        t_vox = float(target_regions[:, i].sum().item())
        results[f"pred_{name}_voxels"] = p_vox
        results[f"target_{name}_voxels"] = t_vox

        # Dice calculation
        if t_vox == 0.0 and p_vox == 0.0:
            d_val = 1.0
        elif t_vox == 0.0 or p_vox == 0.0:
            d_val = 0.0
        else:
            d_val = float(np.nan_to_num(dice_scores[:, i].mean(), nan=0.0))
        results[f"dice_{name}"] = d_val

        # HD95 calculation (penalize missed or false positive structures)
        if t_vox == 0.0 and p_vox == 0.0:
            h_val = 0.0
        elif t_vox == 0.0 or p_vox == 0.0:
            h_val = MAX_BRATS_DISTANCE_MM
        else:
            raw_h = float(hd95_scores[:, i].mean())
            if np.isnan(raw_h) or np.isinf(raw_h):
                h_val = MAX_BRATS_DISTANCE_MM
            else:
                h_val = raw_h
        results[f"hd95_{name}"] = h_val

    return results
