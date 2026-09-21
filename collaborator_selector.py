"""Reproducible sliding-window collaborator selection for Flower strategies.

The selector mirrors the FeTS/RegSimAgg participation idea: only a percentage
of institutions trains in a round, but every institution is selected once before
the next cycle starts.  ``fixed`` remains the default experiment mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from random import Random
from typing import Sequence


@dataclass
class SlidingWindowCollaboratorSelector:
    """Select non-overlapping client windows, reshuffled between cycles."""

    fraction: float = 0.2
    seed: int = 42
    history: list[dict] = field(default_factory=list)
    _client_ids: tuple[int, ...] = field(default_factory=tuple, init=False)
    _windows: list[list[int]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("collaborator selection fraction must be in (0, 1]")

    def _make_windows(self, client_ids: Sequence[int], cycle: int) -> list[list[int]]:
        ordered = list(sorted(client_ids))
        window_size = max(1, int(len(ordered) * self.fraction))
        # The final short window is padded so every round selects the same count.
        # With 33 clients and a six-client window, three clients repeat at the
        # cycle boundary; this preserves the paper-style fixed participation size.
        padding = (-len(ordered)) % window_size
        if padding:
            ordered.extend(ordered[:padding])
        Random(self.seed + cycle).shuffle(ordered)
        return [ordered[start : start + window_size] for start in range(0, len(ordered), window_size)]

    def select(self, client_ids: Sequence[int], server_round: int) -> list[int]:
        """Return the reproducible selected client IDs for a 1-based FL round."""
        ids = tuple(sorted(int(node_id) for node_id in client_ids))
        if not ids:
            return []
        if ids != self._client_ids:
            self._client_ids = ids
            self._windows = self._make_windows(ids, cycle=0)
            self.history.clear()

        round_index = max(server_round - 1, 0)
        windows_per_cycle = len(self._windows)
        cycle, window_index = divmod(round_index, windows_per_cycle)
        if cycle > 0:
            self._windows = self._make_windows(ids, cycle=cycle)

        selected = list(self._windows[window_index])
        self.history.append(
            {
                "server_round": server_round,
                "cycle": cycle,
                "selected_node_ids": selected,
                "selected_count": len(selected),
                "available_count": len(ids),
            }
        )
        return selected
