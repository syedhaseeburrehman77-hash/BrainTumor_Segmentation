"""Pooled-data baseline for comparison only; it deliberately does not use Flower."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from monai.losses import DiceCELoss

from ML_model import build_model
from dataset import make_loaders, read_partitioning


def main(data_root: str, partition_csv: str, epochs: int, batch_size: int, output: str, device: str = "auto") -> None:
    records = [record for _, group in read_partitioning(data_root, partition_csv) for record in group]
    trainloader, _ = make_loaders(records, batch_size=batch_size)
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            runtime_device = torch.device("cuda:0")
            print(f"[Device] Using CUDA GPU: '{torch.cuda.get_device_name(0)}'")
        else:
            runtime_device = torch.device("cpu")
            print("[Device] CUDA requested but not found. Using CPU.")
    else:
        runtime_device = torch.device("cpu")
        print("[Device] Using CPU.")

    model = build_model().to(runtime_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in trainloader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(runtime_device))
            loss = criterion(logits, batch["label"].to(runtime_device).long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"epoch={epoch + 1}/{epochs} loss={sum(losses) / max(len(losses), 1):.4f}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--partition-csv", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="artifacts/centralized_fets2022.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    main(**vars(parser.parse_args()))
