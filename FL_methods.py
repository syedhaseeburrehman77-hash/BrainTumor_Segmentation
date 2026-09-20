"""Factory for Flower baselines, RegSimAgg, FedIN-EDAR, and optional selection."""

from __future__ import annotations

from flwr.serverapp.strategy import (
    FedAdagrad, FedAdam, FedAvg, FedAvgM, FedMedian, FedProx,
    FedTrimmedAvg, FedYogi, QFedAvg,
)

from fedind_dar_strategy import FedINDARStrategy
from regsimagg_strategy import RegSimAggStrategy
from selection_strategies import (
    SlidingFedAdagrad, SlidingFedAdam, SlidingFedAvg, SlidingFedAvgM,
    SlidingFedINDAR, SlidingFedMedian, SlidingFedProx, SlidingFedTrimmedAvg,
    SlidingFedYogi, SlidingQFedAvg, SlidingRegSimAgg,
)


_FIXED_CLASSES = {
    "fedavg": FedAvg, "fedprox": FedProx, "fedavgm": FedAvgM,
    "fedadagrad": FedAdagrad, "fedadam": FedAdam, "fedyogi": FedYogi,
    "qfedavg": QFedAvg, "fedmedian": FedMedian, "fedtrimmedavg": FedTrimmedAvg,
    "regsimagg": RegSimAggStrategy, "fedindar": FedINDARStrategy,
}
_SLIDING_CLASSES = {
    "fedavg": SlidingFedAvg, "fedprox": SlidingFedProx, "fedavgm": SlidingFedAvgM,
    "fedadagrad": SlidingFedAdagrad, "fedadam": SlidingFedAdam, "fedyogi": SlidingFedYogi,
    "qfedavg": SlidingQFedAvg, "fedmedian": SlidingFedMedian,
    "fedtrimmedavg": SlidingFedTrimmedAvg, "regsimagg": SlidingRegSimAgg,
    "fedindar": SlidingFedINDAR,
}


def _common_settings(num_clients: int, fraction_train: float, fraction_evaluate: float) -> dict:
    return {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "min_train_nodes": num_clients,
        "min_evaluate_nodes": num_clients,
        "min_available_nodes": num_clients,
        "weighted_by_key": "num-examples",
    }


def build_strategy(strategy_name: str, config: dict, num_clients: int):
    """Build a fixed-client strategy or its opt-in sliding-window equivalent."""
    strategy_name = strategy_name.lower()
    if strategy_name not in _FIXED_CLASSES:
        raise ValueError(f"Unsupported strategy '{strategy_name}'")

    selector_mode = str(config.get("collaborator-selector", "fixed")).lower()
    if selector_mode not in {"fixed", "sliding"}:
        raise ValueError("collaborator-selector must be 'fixed' or 'sliding'")
    selected = selector_mode == "sliding"
    strategy_class = (_SLIDING_CLASSES if selected else _FIXED_CLASSES)[strategy_name]

    fraction_train = float(config.get("fraction-train", 1.0))
    fraction_evaluate = float(config.get("fraction-evaluate", 1.0))
    settings = _common_settings(num_clients, fraction_train, fraction_evaluate)
    if selected:
        settings.update({
            "collaborator_selection_fraction": float(config.get("collaborator-fraction", 0.2)),
            "collaborator_selection_seed": int(config.get("collaborator-selector-seed", 42)),
        })

    learning_rate = float(config.get("learning-rate", 1e-4))
    server_learning_rate = float(config.get("server-learning-rate", 1.0))
    fedopt_eta = float(config.get("fedopt-eta", 0.1))
    tau = float(config.get("fedopt-tau", 1e-3))

    if strategy_name == "fedavg":
        return strategy_class(**settings)
    if strategy_name == "fedprox":
        return strategy_class(**settings, proximal_mu=float(config.get("proximal-mu", 0.01)))
    if strategy_name == "fedavgm":
        return strategy_class(**settings, server_learning_rate=server_learning_rate,
                              server_momentum=float(config.get("server-momentum", 0.9)))
    if strategy_name == "fedadagrad":
        return strategy_class(**settings, eta=fedopt_eta, eta_l=learning_rate, tau=tau)
    if strategy_name == "fedadam":
        return strategy_class(**settings, eta=fedopt_eta, eta_l=learning_rate, tau=tau,
                              beta_1=float(config.get("beta-1", 0.9)), beta_2=float(config.get("beta-2", 0.99)))
    if strategy_name == "fedyogi":
        return strategy_class(**settings, eta=fedopt_eta, eta_l=learning_rate, tau=tau,
                              beta_1=float(config.get("beta-1", 0.9)), beta_2=float(config.get("beta-2", 0.99)))
    if strategy_name == "qfedavg":
        return strategy_class(**settings, client_learning_rate=learning_rate,
                              q=float(config.get("qfedavg-q", 0.1)))
    if strategy_name == "fedmedian":
        return strategy_class(**settings)
    if strategy_name == "fedtrimmedavg":
        return strategy_class(**settings, beta=float(config.get("trim-beta", 0.2)))
    if strategy_name == "regsimagg":
        return strategy_class(**settings,
                              regularization_round=int(config.get("regsimagg-regularization-round", 5)),
                              distance_mode=str(config.get("regsimagg-distance-mode", "paper_l1")))
    return strategy_class(
        **settings,
        optimization_rounds=int(config.get("num-server-rounds", 5)),
        base_mu=float(config.get("fedindar-base-mu", 0.01)),
        alpha=float(config.get("fedindar-alpha", 2.0)),
        temporal_beta=float(config.get("fedindar-temporal-beta", 0.5)),
        min_mu=float(config.get("fedindar-min-mu", 0.001)),
        max_mu=float(config.get("fedindar-max-mu", 0.1)),
    )
