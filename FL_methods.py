"""Factory for Flower's built-in strategies (no custom aggregation)."""

from __future__ import annotations

from flwr.serverapp.strategy import (
    FedAdagrad,
    FedAdam,
    FedAvg,
    FedAvgM,
    FedMedian,
    FedProx,
    FedTrimmedAvg,
    FedYogi,
    QFedAvg,
)


def _common_settings(
    num_clients: int,
    fraction_train: float,
    fraction_evaluate: float,
) -> dict:
    return {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "min_train_nodes": num_clients,
        "min_evaluate_nodes": num_clients,
        "min_available_nodes": num_clients,
        "weighted_by_key": "num-examples",
    }


def make_fedavg(num_clients: int, fraction_train: float, fraction_evaluate: float) -> FedAvg:
    """Standard weighted average strategy."""
    return FedAvg(**_common_settings(num_clients, fraction_train, fraction_evaluate))


def make_fedprox(num_clients: int, fraction_train: float, fraction_evaluate: float, proximal_mu: float) -> FedProx:
    """FedProx: prevents local models drifting too far on heterogeneous data."""
    return FedProx(
        **_common_settings(num_clients, fraction_train, fraction_evaluate),
        proximal_mu=proximal_mu,
    )


def make_fedavgm(
    num_clients: int,
    fraction_train: float,
    fraction_evaluate: float,
    server_lr: float,
    momentum: float,
) -> FedAvgM:
    """FedAvgM: FedAvg with server-side momentum."""
    return FedAvgM(
        **_common_settings(num_clients, fraction_train, fraction_evaluate),
        server_learning_rate=server_lr,
        server_momentum=momentum,
    )


def make_fedadagrad(
    num_clients: int,
    fraction_train: float,
    fraction_evaluate: float,
    eta: float,
    eta_l: float,
    tau: float,
) -> FedAdagrad:
    """FedAdagrad: adaptive server optimization based on Adagrad."""
    return FedAdagrad(
        **_common_settings(num_clients, fraction_train, fraction_evaluate),
        eta=eta,
        eta_l=eta_l,
        tau=tau,
    )


def make_fedadam(
    num_clients: int,
    fraction_train: float,
    fraction_evaluate: float,
    eta: float,
    eta_l: float,
    beta_1: float,
    beta_2: float,
    tau: float,
) -> FedAdam:
    """FedAdam: Adam-like adaptive server optimization."""
    return FedAdam(
        **_common_settings(num_clients, fraction_train, fraction_evaluate),
        eta=eta,
        eta_l=eta_l,
        beta_1=beta_1,
        beta_2=beta_2,
        tau=tau,
    )


def make_fedyogi(
    num_clients: int,
    fraction_train: float,
    fraction_evaluate: float,
    eta: float,
    eta_l: float,
    beta_1: float,
    beta_2: float,
    tau: float,
) -> FedYogi:
    """FedYogi: Yogi-like adaptive server optimization."""
    return FedYogi(
        **_common_settings(num_clients, fraction_train, fraction_evaluate),
        eta=eta,
        eta_l=eta_l,
        beta_1=beta_1,
        beta_2=beta_2,
        tau=tau,
    )


def make_qfedavg(
    num_clients: int,
    fraction_train: float,
    fraction_evaluate: float,
    client_lr: float,
    q: float,
) -> QFedAvg:
    """QFedAvg: fairness-oriented strategy weighting poorly performing clients."""
    return QFedAvg(
        client_learning_rate=client_lr,
        q=q,
        **_common_settings(num_clients, fraction_train, fraction_evaluate),
    )


def make_fedmedian(
    num_clients: int,
    fraction_train: float,
    fraction_evaluate: float,
) -> FedMedian:
    """FedMedian: coordinate-wise median aggregation, robust to outliers."""
    return FedMedian(
        **_common_settings(num_clients, fraction_train, fraction_evaluate),
    )


def make_fedtrimmedavg(
    num_clients: int,
    fraction_train: float,
    fraction_evaluate: float,
    beta: float,
) -> FedTrimmedAvg:
    """FedTrimmedAvg: removes extreme updates before averaging."""
    return FedTrimmedAvg(
        **_common_settings(num_clients, fraction_train, fraction_evaluate),
        beta=beta,
    )

