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

from ML_model import build_model
from FL_methods import (
    make_fedadagrad,
    make_fedadam,
    make_fedavg,
    make_fedavgm,
    make_fedmedian,
    make_fedprox,
    make_fedtrimmedavg,
    make_fedyogi,
    make_qfedavg,
)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    num_clients = int(config["num-clients"])
    strategy_name = str(config["strategy"]).lower()
    fraction_train = float(config.get("fraction-train", 1.0))
    fraction_evaluate = float(config.get("fraction-evaluate", 1.0))
    learning_rate = float(config.get("learning-rate", 1e-4))
    server_learning_rate = float(config.get("server-learning-rate", 1.0))

    if strategy_name == "fedavg":
        strategy = make_fedavg(num_clients, fraction_train, fraction_evaluate)
    elif strategy_name == "fedprox":
        strategy = make_fedprox(
            num_clients, fraction_train, fraction_evaluate,
            float(config.get("proximal-mu", 0.01)),
        )
    elif strategy_name == "fedavgm":
        strategy = make_fedavgm(
            num_clients, fraction_train, fraction_evaluate,
            server_learning_rate, float(config.get("server-momentum", 0.9)),
        )
    elif strategy_name == "fedadagrad":
        strategy = make_fedadagrad(
            num_clients, fraction_train, fraction_evaluate,
            server_learning_rate, learning_rate, float(config.get("fedopt-tau", 1e-3)),
        )
    elif strategy_name == "fedadam":
        strategy = make_fedadam(
            num_clients, fraction_train, fraction_evaluate,
            server_learning_rate, learning_rate,
            float(config.get("beta-1", 0.9)), float(config.get("beta-2", 0.99)),
            float(config.get("fedopt-tau", 1e-3)),
        )
    elif strategy_name == "fedyogi":
        strategy = make_fedyogi(
            num_clients, fraction_train, fraction_evaluate,
            server_learning_rate, learning_rate,
            float(config.get("beta-1", 0.9)), float(config.get("beta-2", 0.99)),
            float(config.get("fedopt-tau", 1e-3)),
        )
    elif strategy_name == "qfedavg":
        strategy = make_qfedavg(
            num_clients, fraction_train, fraction_evaluate,
            learning_rate, float(config.get("qfedavg-q", 0.1)),
        )
    elif strategy_name == "fedmedian":
        strategy = make_fedmedian(num_clients, fraction_train, fraction_evaluate)
    elif strategy_name == "fedtrimmedavg":
        strategy = make_fedtrimmedavg(
            num_clients, fraction_train, fraction_evaluate,
            float(config.get("trim-beta", 0.2)),
        )
    else:
        raise ValueError(
            f"Unsupported strategy '{strategy_name}'. Choose from: "
            "'fedavg', 'fedprox', 'fedavgm', 'fedadagrad', 'fedadam', "
            "'fedyogi', 'qfedavg', 'fedmedian', 'fedtrimmedavg'."
        )

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

