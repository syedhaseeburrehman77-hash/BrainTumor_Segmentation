"""Factory for Flower's built-in FedAvg and FedProx strategies (no custom aggregation)."""

from flwr.serverapp.strategy import FedAvg, FedProx


def make_fedavg(num_clients: int, fraction_train: float, fraction_evaluate: float) -> FedAvg:
    """Factory for Flower's built-in FedAvg strategy."""
    return FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=num_clients,
        min_evaluate_nodes=num_clients,
        min_available_nodes=num_clients,
        weighted_by_key="num-examples",
    )


def make_fedprox(num_clients: int, fraction_train: float, fraction_evaluate: float, proximal_mu: float) -> FedProx:
    """Factory for Flower's built-in FedProx strategy."""
    return FedProx(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=num_clients,
        min_evaluate_nodes=num_clients,
        min_available_nodes=num_clients,
        weighted_by_key="num-examples",
        proximal_mu=proximal_mu,
    )
