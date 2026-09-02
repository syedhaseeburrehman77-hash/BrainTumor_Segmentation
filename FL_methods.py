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


def build_strategy(strategy_name: str, config: dict, num_clients: int):
    """Central factory dispatcher: builds the configured strategy from run_config."""
    strategy_name = strategy_name.lower()
    fraction_train = float(config.get("fraction-train", 1.0))
    fraction_evaluate = float(config.get("fraction-evaluate", 1.0))
    learning_rate = float(config.get("learning-rate", 1e-4))
    server_learning_rate = float(config.get("server-learning-rate", 1.0))
    fedopt_eta = float(config.get("fedopt-eta", 0.1))

    if strategy_name == "fedavg":
        return make_fedavg(num_clients, fraction_train, fraction_evaluate)
    elif strategy_name == "fedprox":
        return make_fedprox(num_clients, fraction_train, fraction_evaluate, float(config.get("proximal-mu", 0.01)))
    elif strategy_name == "fedavgm":
        return make_fedavgm(num_clients, fraction_train, fraction_evaluate, server_learning_rate, float(config.get("server-momentum", 0.9)))
    elif strategy_name == "fedadagrad":
        return make_fedadagrad(num_clients, fraction_train, fraction_evaluate, fedopt_eta, learning_rate, float(config.get("fedopt-tau", 1e-3)))
    elif strategy_name == "fedadam":
        return make_fedadam(num_clients, fraction_train, fraction_evaluate, fedopt_eta, learning_rate, float(config.get("beta-1", 0.9)), float(config.get("beta-2", 0.99)), float(config.get("fedopt-tau", 1e-3)))
    elif strategy_name == "fedyogi":
        return make_fedyogi(num_clients, fraction_train, fraction_evaluate, fedopt_eta, learning_rate, float(config.get("beta-1", 0.9)), float(config.get("beta-2", 0.99)), float(config.get("fedopt-tau", 1e-3)))
    elif strategy_name == "qfedavg":

        return make_qfedavg(num_clients, fraction_train, fraction_evaluate, learning_rate, float(config.get("qfedavg-q", 0.1)))
    elif strategy_name == "fedmedian":
        return make_fedmedian(num_clients, fraction_train, fraction_evaluate)
    elif strategy_name == "fedtrimmedavg":
        return make_fedtrimmedavg(num_clients, fraction_train, fraction_evaluate, float(config.get("trim-beta", 0.2)))
    else:
        raise ValueError(
            f"Unsupported strategy '{strategy_name}'. Choose from: "
            "'fedavg', 'fedprox', 'fedavgm', 'fedadagrad', 'fedadam', "
            "'fedyogi', 'qfedavg', 'fedmedian', 'fedtrimmedavg'."
        )


