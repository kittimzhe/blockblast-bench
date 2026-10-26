"""Uniform random baseline: picks a legal move at random."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

from ..env import Observation
from .base import Action, Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed) if seed is not None else None
        self._moves = 0

    def reset(self, seed: int) -> None:
        # Combine with a fixed offset so agent RNG streams differ from the env's.
        self._rng = random.Random(seed ^ 0x5EED)
        self._moves = 0

    def act(self, obs: Observation, feedback: Optional[str] = None) -> Action:
        assert self._rng is not None, "reset() not called"
        legal = obs.legal_actions()
        piece, row, col = self._rng.choice(legal)
        self._moves += 1
        return Action(piece, row, col)

    @property
    def stats(self) -> Dict[str, Any]:
        return {"moves": self._moves}
