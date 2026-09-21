"""FedIN-EDAR: local InstanceNorm plus EMD and temporal adaptive FedProx."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from ML_model import instance_norm_state_keys

PROFILE_BINS = 10
REGIONS = ("wt", "tc", "et")
_EPS = 1e-12

# Tumour burden is a volume fraction in [0, 1]. Finite edges allow EMD to
# respect the unequal widths of these log-spaced bins.
TUMOR_BURDEN_BIN_EDGES = np.asarray(
    [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1.0],
    dtype=np.float64,
)


def profile_metric_key(region: str, bin_index: int) -> str:
    return f"tumor-profile-{region}-{bin_index}"


def _emd(histogram_a: np.ndarray, histogram_b: np.ndarray) -> float:
    """Normalized 1-D Wasserstein/EMD for the configured unequal-width bins."""
    if histogram_a.shape != (PROFILE_BINS,) or histogram_b.shape != (PROFILE_BINS,):
        raise ValueError(f"Expected {PROFILE_BINS}-bin tumour-burden histograms")
    cumulative_difference = np.cumsum(histogram_a - histogram_b)[:-1]
    bin_widths = np.diff(TUMOR_BURDEN_BIN_EDGES)
    return float(np.sum(np.abs(cumulative_difference) * bin_widths) /
                 (TUMOR_BURDEN_BIN_EDGES[-1] - TUMOR_BURDEN_BIN_EDGES[0]))


class FedINDARStrategy(FedAvg):
    """FedIN with EMD-initialized, per-round dynamic proximal penalties."""

    def __init__(
        self,
        *,
        optimization_rounds: int,
        base_mu: float = 0.01,
        alpha: float = 2.0,
        temporal_beta: float = 0.5,
        min_mu: float = 0.001,
        max_mu: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if (optimization_rounds < 1 or base_mu < 0 or alpha < 0 or temporal_beta < 0
                or min_mu < 0 or max_mu < min_mu):
            raise ValueError("Invalid FedIN-EDAR penalty configuration")
        self.optimization_rounds = optimization_rounds
        self.base_mu = base_mu
        self.alpha = alpha
        self.temporal_beta = temporal_beta
        self.min_mu = min_mu
        self.max_mu = max_mu
        self.local_in_keys = instance_norm_state_keys()
        self.mu_by_node: dict[int, float] = {}
        self.profile_mu_by_node: dict[int, float] = {}
        self.profile_audit: list[dict[str, Any]] = []
        self.round_audit: list[dict[str, Any]] = []
        self._global_state: dict[str, torch.Tensor] | None = None

    def _messages_with_client_config(
        self, base_messages: Iterable[Message], arrays: ArrayRecord,
        config: ConfigRecord, profile_only: bool,
    ) -> list[Message]:
        messages: list[Message] = []
        for base in base_messages:
            node_id = base.metadata.dst_node_id
            if not profile_only and node_id not in self.mu_by_node:
                raise RuntimeError(
                    f"FedIN-EDAR cannot train node {node_id}: its profile is missing. Restart the run."
                )
            node_config = ConfigRecord(dict(config))
            node_config["fedindar-enabled"] = True
            node_config["fedindar-profile-only"] = profile_only
            node_config["proximal-mu"] = float(self.mu_by_node.get(node_id, self.base_mu))
            messages.append(Message(
                content=RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: node_config}),
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
            ))
        return messages

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._global_state = arrays.to_torch_state_dict()
        messages = super().configure_train(server_round, arrays, config, grid)
        return self._messages_with_client_config(messages, arrays, config, profile_only=(server_round == 1))

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if server_round == 1:
            return []
        base_messages = list(super().configure_evaluate(server_round, arrays, config, grid))
        final_test = server_round == self.optimization_rounds + 1
        messages: list[Message] = []
        for base in base_messages:
            node_config = ConfigRecord(dict(config))
            node_config["fedindar-enabled"] = True
            node_config["fedindar-final-test"] = final_test
            messages.append(Message(
                content=RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: node_config}),
                message_type=MessageType.EVALUATE,
                dst_node_id=base.metadata.dst_node_id,
            ))
        return messages

    def _aggregate_profile_round(self, replies: list[Message]) -> tuple[ArrayRecord, MetricRecord]:
        counts = np.asarray([float(r.content["metrics"][self.weighted_by_key]) for r in replies])
        weights = counts / max(counts.sum(), _EPS)
        profiles = [{
            region: np.asarray([float(r.content["metrics"][profile_metric_key(region, i)]) for i in range(PROFILE_BINS)])
            for region in REGIONS
        } for r in replies]
        global_profile = {
            region: sum(weight * profile[region] for weight, profile in zip(weights, profiles))
            for region in REGIONS
        }
        global_profile = {region: values / max(values.sum(), _EPS) for region, values in global_profile.items()}
        rows = []
        for reply, count, weight, profile in zip(replies, counts, weights, profiles):
            distance = float(np.mean([_emd(profile[region], global_profile[region]) for region in REGIONS]))
            mu = float(np.clip(self.base_mu * np.exp(self.alpha * distance), self.min_mu, self.max_mu))
            node_id = reply.metadata.src_node_id
            self.mu_by_node[node_id] = mu
            self.profile_mu_by_node[node_id] = mu
            rows.append({
                "node_id": str(node_id), "num_examples": float(count), "sample_weight": float(weight),
                "emd_distance": distance, "profile_mu": mu,
                "histograms": {region: profile[region].tolist() for region in REGIONS},
            })
        self.profile_audit = [{"global_histograms": {r: v.tolist() for r, v in global_profile.items()}, "clients": rows}]
        if self._global_state is None:
            raise RuntimeError("FedIN-EDAR profile round has no initial global state")
        return ArrayRecord(self._global_state), MetricRecord({"profile_phase": 1.0, "num-examples": float(counts.sum())})

    def _shared_update_divergence(self, states: list[dict[str, torch.Tensor]]) -> np.ndarray:
        """Mean absolute shared-weight update versus the global model sent this round."""
        if self._global_state is None:
            raise RuntimeError("FedIN-EDAR has no global model for update divergence")
        divergences = []
        for state in states:
            absolute_sum, elements = 0.0, 0
            for name, global_value in self._global_state.items():
                if name in self.local_in_keys or not torch.is_floating_point(global_value):
                    continue
                difference = state[name].detach().cpu().to(torch.float64) - global_value.detach().cpu().to(torch.float64)
                absolute_sum += float(difference.abs().sum().item())
                elements += difference.numel()
            divergences.append(absolute_sum / max(elements, 1))
        return np.asarray(divergences, dtype=np.float64)

    def _next_round_penalties(self, replies: list[Message], divergences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set next-round mu using relative update instability, anchored to profile mu."""
        mean, std = float(divergences.mean()), float(divergences.std())
        normalized = np.zeros_like(divergences) if std <= _EPS else np.clip((divergences - mean) / std, -3.0, 3.0)
        sent_mu = np.asarray([self.mu_by_node[r.metadata.src_node_id] for r in replies])
        profile_mu = np.asarray([self.profile_mu_by_node[r.metadata.src_node_id] for r in replies])
        next_mu = np.clip(profile_mu * np.exp(self.temporal_beta * normalized), self.min_mu, self.max_mu)
        for reply, mu in zip(replies, next_mu):
            self.mu_by_node[reply.metadata.src_node_id] = float(mu)
        return sent_mu, normalized, next_mu

    def _aggregate_shared_layers(self, server_round: int, replies: list[Message]) -> ArrayRecord:
        states = [r.content[self.arrayrecord_key].to_torch_state_dict() for r in replies]
        counts = np.asarray([float(r.content["metrics"][self.weighted_by_key]) for r in replies])
        weights = counts / max(counts.sum(), _EPS)
        divergences = self._shared_update_divergence(states)
        sent_mu, normalized, next_mu = self._next_round_penalties(replies, divergences)
        reference = self._global_state or states[0]
        aggregated: dict[str, torch.Tensor] = {}
        for name, reference_value in reference.items():
            if name in self.local_in_keys or not torch.is_floating_point(reference_value):
                aggregated[name] = reference_value.detach().cpu().clone()
            else:
                value = sum(float(w) * state[name].detach().cpu().to(torch.float64) for state, w in zip(states, weights))
                aggregated[name] = value.to(dtype=reference_value.dtype)
        self.round_audit.append({
            "flower_round": server_round, "optimization_round": server_round - 1,
            "node_ids": [str(r.metadata.src_node_id) for r in replies], "num_examples": counts.tolist(),
            "aggregation_weights": weights.tolist(),
            "profile_mu": [float(self.profile_mu_by_node[r.metadata.src_node_id]) for r in replies],
            "proximal_mu_sent": sent_mu.tolist(), "update_divergence": divergences.tolist(),
            "normalized_update_divergence": normalized.tolist(), "proximal_mu_next_round": next_mu.tolist(),
            "private_instance_norm_keys": sorted(self.local_in_keys),
        })
        return ArrayRecord(aggregated)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid:
            return None, None
        if server_round == 1:
            return self._aggregate_profile_round(valid)
        arrays = self._aggregate_shared_layers(server_round, valid)
        return arrays, self.train_metrics_aggr_fn([reply.content for reply in valid], self.weighted_by_key)
