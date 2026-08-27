"""Run a CPU-first local Flower simulation on a selected FeTS client subset."""

from __future__ import annotations

import os
import sys

# Suppress unwanted C++ logging and disable Windows Job Object restrictions for Ray
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RAY_ENABLE_WINDOWS_JOB_OBJECT"] = "0"
sys.modules.setdefault("tensorflow", None)

import argparse
from pathlib import Path
import subprocess

from verify_dataset import verify_dataset


def patch_all_ray_installations():
    """Locate and patch all ray/_private/utils.py files to prevent AssignProcessToJobObject crashes on Windows."""
    search_dirs = [
        Path(sys.executable).parent,
        Path.home() / "AppData" / "Local" / "uv" / "cache",
        Path.home() / ".flwr" / "runtime-envs",
        Path.home() / "AppData" / "Local" / "Packages",
    ]
    target_snippet = 'raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject() failed")'
    replacement_snippet = 'pass  # Suppress Windows Job Object error'

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        try:
            for utils_path in base_dir.rglob("utils.py"):
                if "ray" in str(utils_path) and "_private" in str(utils_path):
                    try:
                        content = utils_path.read_text(encoding="utf-8", errors="ignore")
                        if target_snippet in content:
                            new_content = content.replace(target_snippet, replacement_snippet)
                            utils_path.write_text(new_content, encoding="utf-8")
                    except Exception:
                        pass
        except Exception:
            pass


def cleanup_stale_flower_processes():
    """Ensure no zombie SuperLink or Ray processes hold port 39093 or state locks on Windows."""
    if sys.platform == "win32":
        for proc in ("flower-superlink.exe", "flower-simulation.exe", "ray.exe", "raylet.exe"):
            try:
                subprocess.run(["taskkill", "/F", "/IM", proc, "/T"], capture_output=True, check=False)
            except Exception:
                pass

    superlink_dir = Path.home() / ".flwr" / "local-superlink"
    if superlink_dir.exists():
        try:
            import shutil
            shutil.rmtree(superlink_dir, ignore_errors=True)
        except Exception:
            pass


def main(clients: int, rounds: int, strategy: str, cpus_per_client: int, device: str = "auto") -> int:
    if clients < 1 or rounds < 1 or cpus_per_client < 1:
        raise ValueError("clients, rounds, and cpus-per-client must all be positive")
    
    cleanup_stale_flower_processes()
    patch_all_ray_installations()
    project_dir = Path(__file__).resolve().parent
    verify_dataset(project_dir / "pyproject.toml", requested_clients=clients)

    # Detect CUDA GPU availability first; fall back to CPU if unavailable
    import torch
    cuda_available = torch.cuda.is_available()
    if device == "cuda" or (device == "auto" and cuda_available):
        if not cuda_available:
            print("[Hardware Detection] CUDA was explicitly requested but no NVIDIA GPU was found. Falling back to CPU.")
            selected_device = "cpu"
            gpus_per_client = 0.0
        else:
            device_name = torch.cuda.get_device_name(0)
            print(f"[Hardware Detection] CUDA GPU detected: '{device_name}'. Acceleration ENABLED.")
            selected_device = "cuda"
            gpus_per_client = 1.0
    else:
        print("[Hardware Detection] CUDA not available (or CPU selected). Acceleration: CPU.")
        selected_device = "cpu"
        gpus_per_client = 0.0

    # Ensure all possible Python Scripts directories are in PATH for flower-superlink and ray
    import site
    import sysconfig

    script_dirs = []
    for scheme in (None, f"{os.name}_user"):
        try:
            p = sysconfig.get_path("scripts", scheme=scheme) if scheme else sysconfig.get_path("scripts")
            if p:
                script_dirs.append(p)
        except Exception:
            pass

    try:
        if hasattr(site, "USER_BASE") and site.USER_BASE:
            script_dirs.append(str(Path(site.USER_BASE) / "Scripts"))
    except Exception:
        pass

    sys_parent = Path(sys.executable).parent
    script_dirs.extend([str(sys_parent), str(sys_parent / "Scripts")])

    valid_dirs = [d for d in script_dirs if d and Path(d).is_dir()]
    new_path = os.pathsep.join(valid_dirs + [os.environ.get("PATH", "")])
    os.environ["PATH"] = new_path
    env = os.environ.copy()

    run_config = (
        f"num-clients={clients} num-server-rounds={rounds} "
        f'strategy="{strategy}" device="{selected_device}" num-workers=0'
    )
    federation_config = (
        f"num-supernodes={clients} "
        f"client-resources-num-cpus={cpus_per_client} "
        f"client-resources-num-gpus={gpus_per_client}"
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
    parser.add_argument(
        "--strategy",
        choices=(
            "fedavg",
            "fedprox",
            "fedavgm",
            "fedadagrad",
            "fedadam",
            "fedyogi",
            "qfedavg",
            "fedmedian",
            "fedtrimmedavg",
        ),
        default="fedavg",
        help="Federated aggregation strategy to benchmark.",
    )
    parser.add_argument("--cpus-per-client", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="Device to use: 'auto' (checks CUDA first, else CPU), 'cuda', or 'cpu'.")
    arguments = parser.parse_args()
    raise SystemExit(main(**vars(arguments)))



