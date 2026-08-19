"""Fail early if the extracted FeTS training data or CSV is incomplete."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dataset import read_partitioning


def verify_dataset(pyproject_path: str | Path = "pyproject.toml", requested_clients: int | None = None) -> None:
    config_path = Path(pyproject_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    root = Path(config["data-root"])
    csv_path = Path(config["partition-csv"])
    if not root.is_dir():
        raise FileNotFoundError(f"Set data-root in {config_path} to the extracted FeTS training-data directory: {root}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Set partition-csv in {config_path} to partitioning_1.csv: {csv_path}")
    partitions = read_partitioning(root, csv_path)
    clients = int(config["num-clients"]) if requested_clients is None else requested_clients
    if not 1 <= clients <= len(partitions):
        raise ValueError(f"Requested {clients} clients; partitioning CSV contains {len(partitions)} institutions")
    selected_cases = sum(len(records) for _, records in partitions[:clients])
    print(f"Verified a {clients}-institution subset with {selected_cases} labelled FeTS cases (of {len(partitions)} institutions).")


if __name__ == "__main__":
    verify_dataset()
