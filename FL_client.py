from __future__ import annotations

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RAY_ENABLE_WINDOWS_JOB_OBJECT"] = "0"
sys.modules.setdefault("tensorflow", None)

import time
from pathlib import Path

import pandas as pd
import torch
from flwr.clientapp import ClientApp
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict
from monai.inferers import sliding_window_inference

from ML_model import build_model
from dataset import client_records, fets_region_metrics, make_loaders

app = ClientApp()


CLIENT_CSV_COLUMNS = [
    "institution_id",
    "phase",
    "loss",
    "dice_et",
    "dice_tc",
    "dice_wt",
    "hd95_et",
    "hd95_tc",
    "hd95_wt",
    "pred_et_voxels",
    "pred_tc_voxels",
    "pred_wt_voxels",
    "target_et_voxels",
    "target_tc_voxels",
    "target_wt_voxels",
    "time_sec",
    "examples",
]


def _append_client_csv(context: Context, row: dict) -> None:
    output_dir = Path(context.run_config.get("output-dir", "artifacts"))
    strategy = str(context.run_config.get("strategy", "federated")).lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / f"{strategy}_fets2022_client_history.csv"
    
    full_row = {col: row.get(col, "") for col in CLIENT_CSV_COLUMNS}
    df = pd.DataFrame([full_row], columns=CLIENT_CSV_COLUMNS)
    if not csv_file.exists():
        df.to_csv(csv_file, index=False, mode="w")
    else:
        df.to_csv(csv_file, index=False, mode="a", header=False)


def _device(context: Context) -> torch.device:
    """Check GPU/CUDA availability first; use CUDA if available, otherwise fall back to CPU."""
    requested = str(context.run_config.get("device", "auto")).lower()
    if requested in {"cuda", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif requested == "cuda":
            print("[Device Warning] CUDA requested but not found on this system. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def _loaders(context: Context):
    partition_index = int(context.node_config["partition-id"])
    records = client_records(
        context.run_config["data-root"],
        context.run_config["partition-csv"],
        partition_index,
        global_test_fraction=float(context.run_config["global-test-fraction"]),
        seed=int(context.run_config["seed"]),
    )
    device = _device(context)
    return make_loaders(
        records,
        batch_size=int(context.run_config["batch-size"]),
        cache_rate=float(context.run_config["cache-rate"]),
        seed=int(context.run_config["seed"]) + partition_index,
        num_workers=int(context.run_config["num-workers"]),
        pin_memory=(device.type == "cuda"),
    )


def _train_one_client(model, loader, epochs: int, lr: float, proximal_mu: float, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    # FedProx compares locally updated weights with this received global model.
    global_params = [p.detach().clone() for p in model.parameters()]
    total_loss, steps = 0.0, 0
    for _ in range(epochs):
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long().squeeze(1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            if proximal_mu > 0.0:
                proximal_term = sum(torch.sum((p - g) ** 2) for p, g in zip(model.parameters(), global_params))
                loss = loss + 0.5 * proximal_mu * proximal_term
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
    return total_loss / max(steps, 1)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the received global model on exactly one FeTS institution."""
    start_time = time.time()
    partition_index = int(context.node_config["partition-id"])
    device = _device(context)
    model = build_model().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainloader, _ = _loaders(context)
    # The built-in FedProx strategy injects proximal-mu; FedAvg leaves it at zero.
    proximal_mu = float(msg.content["config"].get("proximal-mu", 0.0))
    loss = _train_one_client(
        model, trainloader,
        epochs=int(context.run_config["local-epochs"]),
        lr=float(msg.content["config"]["lr"]),
        proximal_mu=proximal_mu,
        device=device,
    )
    elapsed = time.time() - start_time
    print(
        f"[Institution {partition_index:02d}] Training Finished | "
        f"Loss: {loss:.4f} | Time: {elapsed:.2f}s | "
        f"Examples: {len(trainloader.dataset)}"
    )
    _append_client_csv(context, {
        "institution_id": f"{partition_index:02d}",
        "phase": "train",
        "loss": loss,
        "time_sec": elapsed,
        "examples": len(trainloader.dataset),
    })
    metrics = MetricRecord({
        "train_loss": loss,
        "train_time_sec": elapsed,
        "num-examples": len(trainloader.dataset),
    })
    return Message(
        content=RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics}),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the received global model on the client's held-out labelled cases."""
    start_time = time.time()
    partition_index = int(context.node_config["partition-id"])
    device = _device(context)
    model = build_model().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    _, valloader = _loaders(context)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    totals = {
        "loss": 0.0,
        "dice_et": 0.0, "dice_tc": 0.0, "dice_wt": 0.0,
        "hd95_et": 0.0, "hd95_tc": 0.0, "hd95_wt": 0.0,
        "pred_et_voxels": 0.0, "pred_tc_voxels": 0.0, "pred_wt_voxels": 0.0,
        "target_et_voxels": 0.0, "target_tc_voxels": 0.0, "target_wt_voxels": 0.0,
    }
    with torch.no_grad():
        for batch in valloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            logits = sliding_window_inference(images, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model)
            totals["loss"] += criterion(logits, labels.squeeze(1)).item()
            for key, value in fets_region_metrics(logits, labels).items():
                totals[key] += value
    count = max(len(valloader), 1)
    eval_loss = totals["loss"] / count
    dice_et = totals["dice_et"] / count
    dice_tc = totals["dice_tc"] / count
    dice_wt = totals["dice_wt"] / count
    hd95_et = totals["hd95_et"] / count
    hd95_tc = totals["hd95_tc"] / count
    hd95_wt = totals["hd95_wt"] / count
    pred_wt_vox = totals["pred_wt_voxels"] / count
    target_wt_vox = totals["target_wt_voxels"] / count
    elapsed = time.time() - start_time
    print(
        f"[Institution {partition_index:02d}] Evaluation Finished | "
        f"Loss: {eval_loss:.4f} | "
        f"Dice (ET/TC/WT): {dice_et:.4f} / {dice_tc:.4f} / {dice_wt:.4f} | "
        f"HD95 (ET/TC/WT): {hd95_et:.2f} / {hd95_tc:.2f} / {hd95_wt:.2f} | "
        f"Pred/Target WT Voxels: {pred_wt_vox:.0f} / {target_wt_vox:.0f} | "
        f"Time: {elapsed:.2f}s | Examples: {len(valloader.dataset)}"
    )
    _append_client_csv(context, {
        "institution_id": f"{partition_index:02d}",
        "phase": "evaluate",
        "loss": eval_loss,
        "dice_et": dice_et,
        "dice_tc": dice_tc,
        "dice_wt": dice_wt,
        "hd95_et": hd95_et,
        "hd95_tc": hd95_tc,
        "hd95_wt": hd95_wt,
        "pred_et_voxels": totals["pred_et_voxels"] / count,
        "pred_tc_voxels": totals["pred_tc_voxels"] / count,
        "pred_wt_voxels": pred_wt_vox,
        "target_et_voxels": totals["target_et_voxels"] / count,
        "target_tc_voxels": totals["target_tc_voxels"] / count,
        "target_wt_voxels": target_wt_vox,
        "time_sec": elapsed,
        "examples": len(valloader.dataset),
    })
    metrics = MetricRecord({
        "eval_loss": eval_loss,
        "eval_dice_et": dice_et,
        "eval_dice_tc": dice_tc,
        "eval_dice_wt": dice_wt,
        "eval_hd95_et": hd95_et,
        "eval_hd95_tc": hd95_tc,
        "eval_hd95_wt": hd95_wt,
        "eval_pred_wt_voxels": pred_wt_vox,
        "eval_target_wt_voxels": target_wt_vox,
        "eval_time_sec": elapsed,
        "num-examples": len(valloader.dataset),
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)

