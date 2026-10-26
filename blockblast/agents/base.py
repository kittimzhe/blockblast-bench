"""Agent interface shared by every benchmark participant."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..env import Observation


class AgentError(Exception):
    """Base class for recoverable agent-side failures."""


class AgentParseError(AgentError):
    """The agent produced a reply that cannot be turned into a legal action.

    The harness treats this like an illegal move: it feeds the error back and
    lets the agent retry a bounded number of times.
    """


@dataclass(frozen=True)
class Action:
    """Placement of tray slot ``piece`` with its top-left cell at (row, col)."""

    piece: int
    row: int
    col: int

    def as_tuple(self):
        return (self.piece, self.row, self.col)

    def __iter__(self):
        return iter(self.as_tuple())


class Agent(ABC):
    """A benchmark participant.

    One Agent instance handles exactly one episode; the harness calls
    :meth:`reset` at episode start and :meth:`act` once per move. Agents must
    be re-creatable from a factory for concurrent episode execution.
    """

    name: str = "agent"

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Prepare for a new episode with the given seed."""

    @abstractmethod
    def act(self, obs: Observation, feedback: Optional[str] = None) -> Action:
        """Choose an action for the current observation.

        ``feedback`` is non-None when a previous attempt was rejected
        (illegal move or unparseable reply); the agent should try again.
        """

    @property
    def stats(self) -> Dict[str, Any]:
        """Metrics accumulated during the current episode."""
        return {}

    @property
    def dialog(self) -> list:
        """Optional per-move dialog log (used by --save-dialog for LLMs)."""
        return []
