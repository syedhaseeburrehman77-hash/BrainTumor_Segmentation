# FeTS 2022: Flower Federated Segmentation

This project uses Flower's current ClientApp, ServerApp, ArrayRecord, and
built-in strategies plus an optional RegSimAgg aggregation strategy.

## Setup

Use 64-bit Python 3.11, not Python 3.14. Flower's documentation recommends
Python 3.11 for simulations because of Ray compatibility. The Python 3.14
package resolver can select old source-only packages (including Matplotlib)
that cannot build on Python 3.14.

On Windows PowerShell, create and activate an isolated Python 3.11 environment:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip setuptools wheel

1. Download and extract MICCAI_FeTS2022_TrainingData.zip.
2. Set the two Windows paths in pyproject.toml:
   data-root is the extracted directory containing FeTS2022_### folders;
   
   partition-csv is its partitioning_1.csv.
3. Install. For a CPU-only Windows installation, install the official CPU
   wheel first:

   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt

4. Verify the dataset:

   python verify_dataset.py

5. Run a small CPU-only federated experiment:

   python flower_run.py --clients 3 --rounds 2 --strategy fedavg

The clients argument selects the first N real institution partitions from
partitioning_1.csv, and rounds controls the number of Flower server rounds.
The runner configures CPU-only virtual SuperNodes and then invokes Flower with
streaming logs. Start with one client and one round to validate memory use.

## Strategy selection

strategy = "fedavg" selects Flower's built-in FedAvg.

strategy = "fedprox" selects Flower's built-in FedProx. The client reads
the proximal-mu value that the documented Flower FedProx strategy places in
the training config and adds the required proximal loss term. No aggregation
method is reimplemented.

## Metrics

The client holds out 15% of each institution's labelled cases locally. Each
round reports Dice and HD95 for enhancing tumour (ET), tumour core (TC), and
whole tumour (WT). The Synapse validation archive is not used for metrics
because its ground-truth segmentations are protected.

## Central baseline

For a pooled-data comparison only:

python centralize_baseline.py --data-root "C:/.../TrainingData" --partition-csv "C:/.../partitioning_1.csv"

It must not be used as the federated result.

## Interactive Strategy Menu & Strategies

You can launch the runner interactively:

    python flower_run.py --clients 3 --rounds 5

An interactive numbered menu allows selecting any of Flower's 9 built-in strategies:
1. `fedavg` - Standard weighted average
2. `fedprox` - Heterogeneous non-IID regularizer
3. `fedavgm` - Server-side momentum
4. `fedadagrad` - Adaptive server learning rates
5. `fedadam` - Adam-like server optimization
6. `fedyogi` - Yogi-like adaptive variance control
7. `qfedavg` - Fairness-oriented weighting
8. `fedmedian` - Robust coordinate-wise median
9. `fedtrimmedavg` - Robust trimmed mean
10. `regsimagg` - Opt-in model-similarity and temporal aggregation
11. `fedindar` - Local IN + EMD + Temporal Adaptive FedProx

You can also pass the strategy directly via CLI:

    python flower_run.py --clients 3 --rounds 5 --strategy fedadam

### RegSimAgg (opt-in)

Run it without changing application code:

    python flower_run.py --clients 3 --rounds 12 --strategy regsimagg

RegSimAgg combines each client's normalized sample-size weight with a normalized
model-similarity weight. After `regsimagg-regularization-round` (default `10`),
it also down-weights a client update that has a large mean absolute change from
the preceding global model. `regsimagg-distance-mode = "paper_l1"` is the
recommended full-parameter L1 mode; `"github_compat"` is retained only to
reproduce the supplied prototype's scalar-distance calculation.

It does not transmit images, labels, patient identifiers, or extra client-side
profiles. The server writes `artifacts/regsimagg_aggregation_audit.json`, which
contains the per-round sample, similarity, temporal, and final weights.
Set the start threshold without editing code, for example:

    python flower_run.py --clients 3 --rounds 8 --strategy regsimagg --regsimagg-regularization-round 5 --device cpu

This keeps rounds 1--5 as base RegSimAgg and starts temporal damping at round 6.

### Comparison: RegSimAgg (Paper) vs. FedIN-EDAR (Proposed)

| Feature | RegSimAgg (Paper) | FedIN-EDAR (Proposed) |
|:---|:---|:---|
| **Mechanism** | Server-side similarity and temporal weighting | Client personalization + dynamic client regularization |
| **Personalization** | Shared global model across all clients | Institution-specific private InstanceNorm layers |
| **Heterogeneity Handling** | Model parameter distance to centroid | Earth Mover's Distance (EMD) on regional tumor-burden distributions |
| **Client Regularization** | None (standard local optimization) | Dynamic FedProx with drift-adaptive proximal penalty ($\mu$) |
| **Transmitted Weights** | Entire network | Shared layers only (InstanceNorm remains local) |

## Hardware Auto-Detection (GPU / CPU)

The runner automatically checks for NVIDIA CUDA GPU availability:
* If a CUDA device is detected, training runs on GPU (`cuda:0`).
* If no CUDA device is present, it automatically falls back to CPU.
* Override option: `--device auto`, `--device cuda`, or `--device cpu`.

## Global Test Module

To evaluate the final aggregated model on completely unseen data:
* `global-test-fraction = 0.15` in `pyproject.toml` reserves 15% of cases from each institution.
* Clients only receive the remaining 85% for local training and validation (zero data leakage).
* After all federated rounds finish, the server runs a final evaluation on the reserved global test set.

## Output Artifacts

All training artifacts and logs are saved in `artifacts/`:
* `artifacts/<strategy>_fets2022_final.pt`: Final model weights
* `artifacts/<strategy>_fets2022_metrics.csv`: Round-by-round global federated metrics
* `artifacts/<strategy>_fets2022_client_history.csv`: Per-institution training and validation metrics
* `artifacts/<strategy>_fets2022_global_test.csv`: Final unseen global test results
* `artifacts/regsimagg_aggregation_audit.json`: RegSimAgg per-round aggregation audit

