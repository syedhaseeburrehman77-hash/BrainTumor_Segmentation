"""Run a CPU-first local Flower simulation on a selected FeTS client subset."""

from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path

from verify_dataset import verify_dataset


import os


def main(clients: int, rounds: int, strategy: str, cpus_per_client: int) -> int:
    if clients < 1 or rounds < 1 or cpus_per_client < 1:
        raise ValueError("clients, rounds, and cpus-per-client must all be positive")
    project_dir = Path(__file__).resolve().parent
    verify_dataset(project_dir / "pyproject.toml", requested_clients=clients)

    # Ensure virtual environment Scripts directory is in PATH for flower-superlink and ray
    env = os.environ.copy()
    scripts_dir = str(Path(sys.executable).parent)
    env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")

    run_config = (
        f"num-clients={clients} num-server-rounds={rounds} "
        f'strategy="{strategy}" device="cpu" num-workers=0'
    )
    federation_config = (
        f"num-supernodes={clients} "
        f"client-resources-num-cpus={cpus_per_client} "
        f"client-resources-num-gpus=0.0"
    )

    return subprocess.call(
        [
            sys.executable, "-m", "flwr.cli.app", "run", ".",
            "--stream",
            "--run-config", run_config,
            "--federation-config", federation_config,
        ],
        cwd=project_dir,
        env=env,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=3, help="Use the first N real institution partitions (1-23).")
    parser.add_argument("--rounds", type=int, default=2, help="Number of Flower server rounds.")
    parser.add_argument("--strategy", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--cpus-per-client", type=int, default=1)
    arguments = parser.parse_args()
    raise SystemExit(main(**vars(arguments)))
