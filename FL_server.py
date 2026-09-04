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
from monai.losses import DiceCELoss

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
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    totals = {
        "test_loss": 0.0,
        "dice_et": 0.0, "dice_tc": 0.0, "dice_wt": 0.0,
        "hd95_et": 0.0, "hd95_tc": 0.0, "hd95_wt": 0.0,
        "pred_et_voxels": 0.0, "pred_tc_voxels": 0.0, "pred_wt_voxels": 0.0,
        "target_et_voxels": 0.0, "target_tc_voxels": 0.0, "target_wt_voxels": 0.0,
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            logits = sliding_window_inference(images, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model)
            totals["test_loss"] += criterion(logits, labels).item()
            for key, value in fets_region_metrics(logits, labels).items():
                totals[key] += value


    count = max(len(loader), 1)
    results = {key: value / count for key, value in totals.items()}
    results["test_examples"] = len(loader.dataset)
    output_file = Path(config["output-dir"]) / f"{str(config['strategy']).lower()}_fets2022_global_test.csv"
    pd.DataFrame([results]).to_csv(output_file, index=False)
    print(
        f"[Server] Final Unseen Global Test | Loss: {results['test_loss']:.4f} | "
        f"Dice (ET/TC/WT): {results['dice_et']:.4f} / {results['dice_tc']:.4f} / {results['dice_wt']:.4f} | "
        f"HD95 (ET/TC/WT): {results['hd95_et']:.2f} / {results['hd95_tc']:.2f} / {results['hd95_wt']:.2f} | "
        f"Pred/Target WT Voxels: {results['pred_wt_voxels']:.0f} / {results['target_wt_voxels']:.0f}"
    )
    print(f"[Server] Final unseen global-test results saved to: {output_file}")
    return results


def merge_client_history_csv(output_dir: Path, strategy_name: str) -> None:
    """Merge per-institution temporary client CSVs into one final client history CSV."""
    import shutil
    tmp_dir = output_dir / "_client_metrics_tmp"
    if not tmp_dir.exists():
        return

    csv_files = sorted(tmp_dir.glob(f"{strategy_name}_client_*.csv"))
    if not csv_files:
        return

    dfs = [pd.read_csv(f) for f in csv_files if f.stat().st_size > 0]
    if not dfs:
        return

    merged_df = pd.concat(dfs, ignore_index=True)
    if "num_examples" in merged_df.columns and "phase" in merged_df.columns and "round" in merged_df.columns:
        totals = merged_df.groupby(["round", "phase"])["num_examples"].transform("sum")
        merged_df["aggregation_weight"] = (merged_df["num_examples"] / totals).round(6)

    sort_cols = [c for c in ["round", "phase", "institution_id"] if c in merged_df.columns]
    if sort_cols:
        merged_df.sort_values(by=sort_cols, inplace=True)

    final_csv = output_dir / f"{strategy_name}_fets2022_client_history.csv"
    merged_df.to_csv(final_csv, index=False)
    print(f"[Server] Institutional metrics merged into: {final_csv}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


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

    # Merge per-institution temporary CSV files into final client history CSV
    merge_client_history_csv(output_dir, strategy_name)

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
