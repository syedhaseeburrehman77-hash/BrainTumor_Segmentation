from __future__ import annotations

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RAY_ENABLE_WINDOWS_JOB_OBJECT"] = "0"
sys.modules.setdefault("tensorflow", None)

from pathlib import Path

import pandas as pd
import torch
from flwr.common import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from monai.inferers import sliding_window_inference

from ML_model import build_model
from FL_methods import build_strategy
from dataset import fets_region_metrics, global_test_records, make_global_test_loader

app = ServerApp()


# Global-test module: evaluate the final federated model on cases excluded from every client.
def run_global_test(final_state_dict: dict, config) -> dict[str, float]:
    requested_device = str(config["device"]).lower()
    device = torch.device("cuda:0" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(final_state_dict)
    model.eval()

    records = global_test_records(
        config["data-root"],
        config["partition-csv"],
        global_test_fraction=float(config["global-test-fraction"]),
        seed=int(config["seed"]),
    )
    loader = make_global_test_loader(records, num_workers=int(config["num-workers"]))
    criterion = torch.nn.CrossEntropyLoss()
    totals = {"test_loss": 0.0, "dice_et": 0.0, "dice_tc": 0.0, "dice_wt": 0.0,
              "hd95_et": 0.0, "hd95_tc": 0.0, "hd95_wt": 0.0}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            logits = sliding_window_inference(images, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model)
            totals["test_loss"] += criterion(logits, labels.squeeze(1)).item()
            for key, value in fets_region_metrics(logits, labels).items():
                totals[key] += value

    count = len(loader)
    results = {key: value / count for key, value in totals.items()}
    results["test_examples"] = len(loader.dataset)
    output_file = Path(config["output-dir"]) / f"{str(config['strategy']).lower()}_fets2022_global_test.csv"
    pd.DataFrame([results]).to_csv(output_file, index=False)
    print(f"[Server] Final unseen global-test results saved to: {output_file}")
    return results


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    num_clients = int(config["num-clients"])
    strategy_name = str(config["strategy"]).lower()
    strategy = build_strategy(strategy_name, config, num_clients)


    initial_arrays = ArrayRecord(build_model().state_dict())
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": float(config["learning-rate"])}),
        num_rounds=int(config["num-server-rounds"]),
    )
    output_dir = Path(config["output-dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final global model checkpoint
    model_checkpoint_path = output_dir / f"{strategy_name}_fets2022_final.pt"
    final_state_dict = result.arrays.to_torch_state_dict()
    torch.save(final_state_dict, model_checkpoint_path)
    print(f"\n[Server] Model checkpoint saved to: {model_checkpoint_path}")

    # Global-test module: runs once after all federated rounds and client evaluations are complete.
    run_global_test(final_state_dict, config)

    # Build and save round-by-round metrics to CSV
    rows = []
    rounds = sorted(set(list(result.train_metrics_clientapp.keys()) + list(result.evaluate_metrics_clientapp.keys())))
    for r in rounds:
        row = {"round": r, "strategy": strategy_name}
        if r in result.train_metrics_clientapp:
            for k, v in dict(result.train_metrics_clientapp[r]).items():
                row[k] = v
        if r in result.evaluate_metrics_clientapp:
            for k, v in dict(result.evaluate_metrics_clientapp[r]).items():
                row[k] = v
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = output_dir / f"{strategy_name}_fets2022_metrics.csv"
        df.to_csv(csv_path, index=False)
        print("\n" + "=" * 78)
        print(f"        FEDERATED LEARNING RESULTS SUMMARY ({strategy_name.upper()})")
        print("=" * 78)
        print(df.to_string(index=False))
        print("=" * 78)
        print(f"[Server] Round results saved to CSV: {csv_path}\n")

