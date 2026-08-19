"""Flower ServerApp: choose the documented built-in FedAvg or FedProx strategy."""

from pathlib import Path

import pandas as pd
import torch
from flwr.common import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from ML_model import build_model
from FL_methods import make_fedavg, make_fedprox

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    num_clients = int(config["num-clients"])
    strategy_name = str(config["strategy"]).lower()
    if strategy_name == "fedavg":
        strategy = make_fedavg(num_clients, float(config["fraction-train"]), float(config["fraction-evaluate"]))
    elif strategy_name == "fedprox":
        strategy = make_fedprox(
            num_clients, float(config["fraction-train"]), float(config["fraction-evaluate"]),
            float(config["proximal-mu"]),
        )
    else:
        raise ValueError("strategy must be either 'fedavg' or 'fedprox'")

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
    torch.save(result.arrays.to_torch_state_dict(), model_checkpoint_path)
    print(f"\n[Server] Model checkpoint saved to: {model_checkpoint_path}")

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

