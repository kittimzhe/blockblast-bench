"""Agent registry and factory."""

from __future__ import annotations

from typing import Callable, Dict, Type

from .base import Action, Agent, AgentError, AgentParseError
from .beam_agent import BeamAgent
from .greedy_agent import GreedyAgent
from .heuristic_agent import HeuristicAgent
from .llm_agent import LLMAgent, LLMConfig, parse_action
from .random_agent import RandomAgent

AGENT_REGISTRY: Dict[str, Type[Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "heuristic": HeuristicAgent,
    "beam": BeamAgent,
    "llm": LLMAgent,
}


def agent_names() -> list:
    return sorted(AGENT_REGISTRY)


def make_agent(name: str, **kwargs) -> Agent:
    """Build an agent instance; ``kwargs`` are forwarded to the agent class."""
    try:
        cls = AGENT_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown agent {name!r}; available: {', '.join(agent_names())}"
        ) from None
    return cls(**kwargs)


__all__ = [
    "Action",
    "Agent",
    "AgentError",
    "AgentParseError",
    "AGENT_REGISTRY",
    "agent_names",
    "make_agent",
    "RandomAgent",
    "GreedyAgent",
    "HeuristicAgent",
    "BeamAgent",
    "LLMAgent",
    "LLMConfig",
    "parse_action",
]
