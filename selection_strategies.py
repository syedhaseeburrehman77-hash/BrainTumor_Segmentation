"""Opt-in Flower strategy wrappers for sliding-window collaborator selection."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp import Grid
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
from flwr.supercore import log
from logging import INFO

from collaborator_selector import SlidingWindowCollaboratorSelector
from fedind_dar_strategy import FedINDARStrategy
from regsimagg_strategy import RegSimAggStrategy


class _SlidingWindowTrainMixin:
    """Replace Flower random sampling with the reproducible FeTS client cycle."""

    def __init__(
        self,
        *args: Any,
        collaborator_selection_fraction: float = 0.2,
        collaborator_selection_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.collaborator_selector = SlidingWindowCollaboratorSelector(
            fraction=collaborator_selection_fraction,
            seed=collaborator_selection_seed,
        )
        super().__init__(*args, **kwargs)

    @property
    def collaborator_selection_audit(self) -> list[dict]:
        return self.collaborator_selector.history

    def _selected_train_messages(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid,
        selection_round: int | None = None,
    ) -> list[Message]:
        node_ids = list(grid.get_node_ids())
        selected = self.collaborator_selector.select(
            node_ids, server_round if selection_round is None else selection_round
        )
        if not selected:
            return []
        config = ConfigRecord(dict(config))
        config["server-round"] = server_round
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        log(
            INFO,
            "configure_train: Sliding-window selected %s nodes (out of %s): %s",
            len(selected),
            len(node_ids),
            selected,
        )
        return list(self._construct_messages(record, selected, MessageType.TRAIN))

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        return self._selected_train_messages(server_round, arrays, config, grid)


class SlidingFedAvg(_SlidingWindowTrainMixin, FedAvg):
    pass


class SlidingFedProx(_SlidingWindowTrainMixin, FedProx):
    pass


class SlidingFedAvgM(_SlidingWindowTrainMixin, FedAvgM):
    pass


class SlidingFedAdagrad(_SlidingWindowTrainMixin, FedAdagrad):
    pass


class SlidingFedAdam(_SlidingWindowTrainMixin, FedAdam):
    pass


class SlidingFedYogi(_SlidingWindowTrainMixin, FedYogi):
    pass


class SlidingQFedAvg(_SlidingWindowTrainMixin, QFedAvg):
    pass


class SlidingFedMedian(_SlidingWindowTrainMixin, FedMedian):
    pass


class SlidingFedTrimmedAvg(_SlidingWindowTrainMixin, FedTrimmedAvg):
    pass


class SlidingRegSimAgg(_SlidingWindowTrainMixin, RegSimAggStrategy):
    pass


class SlidingFedINDAR(_SlidingWindowTrainMixin, FedINDARStrategy):
    """FedIN-EDAR profiles all hospitals once, then selects a window per round."""

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._global_state = arrays.to_torch_state_dict()
        if server_round == 1:
            # Every client needs a profile before a future selected round can use mu_k.
            base_messages = list(FedAvg.configure_train(self, server_round, arrays, config, grid))
        else:
            # Round 2 is optimization round 1 because round 1 was profiling.
            base_messages = self._selected_train_messages(
                server_round, arrays, config, grid, selection_round=server_round - 1
            )
        return self._messages_with_client_config(
            base_messages, arrays, config, profile_only=(server_round == 1)
        )
