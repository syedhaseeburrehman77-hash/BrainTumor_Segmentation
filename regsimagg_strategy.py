"""Opt-in RegSimAgg aggregation for Flower ServerApp simulations.

The strategy combines normalized sample-size and model-similarity weights. From
``regularization_round + 1`` onward, sharp temporal updates are down-weighted.
Only server-side model arrays are used; no additional client data is sent.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from flwr.common import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

_EPS = 1e-12


def _state_from_reply(reply: Message, arrayrecord_key: str) -> OrderedDict[str, np.ndarray]:
    return OrderedDict(
        (name, tensor.detach().cpu().numpy().copy())
        for name, tensor in reply.content[arrayrecord_key].to_torch_state_dict().items()
    )


def _float_vector(state: dict[str, np.ndarray]) -> np.ndarray:
    parts = [v.astype(np.float64, copy=False).ravel() for v in state.values() if np.issubdtype(v.dtype, np.floating)]
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float64)


def _similarity_weights(states: list[dict[str, np.ndarray]], mode: str) -> tuple[np.ndarray, np.ndarray]:
    vectors = [_float_vector(state) for state in states]
    if mode == "github_compat":
        totals = np.asarray([v.sum() for v in vectors], dtype=np.float64)
        distances = np.abs(totals - totals.mean())
    else:
        centroid = np.mean(np.stack(vectors), axis=0)
        distances = np.asarray([np.mean(np.abs(v - centroid)) for v in vectors])
    inverse = 1.0 / (distances + _EPS)
    return inverse / inverse.sum(), distances


def _aggregate_states(states: list[dict[str, np.ndarray]], weights: np.ndarray) -> OrderedDict[str, torch.Tensor]:
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, first in states[0].items():
        if np.issubdtype(first.dtype, np.floating):
            value = sum(weight * state[name].astype(np.float64, copy=False) for state, weight in zip(states, weights))
            result[name] = torch.from_numpy(value.astype(first.dtype, copy=False))
        else:
            result[name] = torch.from_numpy(first.copy())
    return result


class RegSimAggStrategy(FedAvg):
    """Sample-size + model-similarity aggregation with temporal damping."""

    def __init__(self, *, regularization_round: int = 10, distance_mode: str = "paper_l1", weighted_by_key: str = "num-examples", **kwargs: Any) -> None:
        if regularization_round < 0:
            raise ValueError("regularization_round must be non-negative")
        if distance_mode not in {"paper_l1", "github_compat"}:
            raise ValueError("distance_mode must be 'paper_l1' or 'github_compat'")
        super().__init__(weighted_by_key=weighted_by_key, **kwargs)
        self.regularization_round = regularization_round
        self.distance_mode = distance_mode
        self.weighted_by_key = weighted_by_key
        self.previous_global: OrderedDict[str, np.ndarray] | None = None
        self.aggregation_audit: list[dict[str, Any]] = []

    def aggregate_train(self, server_round: int, replies: list[Message]) -> tuple[ArrayRecord | None, MetricRecord]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, MetricRecord()
        states = [_state_from_reply(reply, self.arrayrecord_key) for reply in valid_replies]
        counts = np.asarray([float(reply.content["metrics"][self.weighted_by_key]) for reply in valid_replies], dtype=np.float64)
        sample_weights = counts / max(counts.sum(), _EPS)
        similarity_weights, distances = _similarity_weights(states, self.distance_mode)
        final_weights = 0.5 * (sample_weights + similarity_weights)
        temporal_change = np.full(len(states), np.nan)
        if server_round > self.regularization_round and self.previous_global is not None:
            temporal_change = np.asarray([np.mean(np.abs(_float_vector(s) - _float_vector(self.previous_global))) for s in states])
            final_weights = final_weights / (temporal_change + _EPS)
        final_weights = final_weights / final_weights.sum()
        aggregated_state = _aggregate_states(states, final_weights)
        self.previous_global = OrderedDict((n, v.detach().cpu().numpy().copy()) for n, v in aggregated_state.items())
        self.aggregation_audit.append({"round": server_round, "node_ids": [str(r.metadata.src_node_id) for r in valid_replies], "num_examples": counts.tolist(), "sample_weights": sample_weights.tolist(), "distances": distances.tolist(), "similarity_weights": similarity_weights.tolist(), "temporal_change": temporal_change.tolist(), "aggregation_weights": final_weights.tolist()})
        return ArrayRecord(aggregated_state), aggregate_metricrecords(
            [reply.content for reply in valid_replies], self.weighted_by_key
        )
