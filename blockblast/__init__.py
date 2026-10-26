"""Block Blast benchmark: game environment + AI agent evaluation harness.

Core design goals
-----------------
* Faithful mechanics of the mobile game Block Blast! (8x8 board, tray of 3
  non-rotatable pieces, full-row/column clears, combos, endless scoring).
* Fully deterministic given a seed -> fair comparison between agents.
* Zero third-party dependencies (stdlib only) for the game core and the
  classic (non-LLM) agents, so the benchmark runs anywhere.
"""

from .env import (
    Action,
    BlockBlastEnv,
    InvalidActionError,
    Observation,
    StepResult,
    can_place,
    clear_lines,
    full_lines,
    legal_actions_for,
    place_cells,
)
from .pieces import SHAPE_CATALOG, Shape
from .scoring import ScoringConfig, compute_move_points

__version__ = "0.1.0"

__all__ = [
    "Action",
    "BlockBlastEnv",
    "InvalidActionError",
    "Observation",
    "StepResult",
    "ScoringConfig",
    "compute_move_points",
    "SHAPE_CATALOG",
    "Shape",
    "can_place",
    "clear_lines",
    "full_lines",
    "legal_actions_for",
    "place_cells",
    "__version__",
]
