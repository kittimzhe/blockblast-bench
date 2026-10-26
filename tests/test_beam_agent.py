"""Tests for the beam-search baseline agent."""

import pytest

from blockblast.agents import BeamAgent, GreedyAgent, HeuristicAgent, make_agent
from blockblast.harness import EvalConfig, run_episode

SEEDS = (2024, 2026)  # two-seed sample keeps the suite fast (~15s)


def _mean(agent_factory, seeds):
    scores = [
        run_episode(agent_factory(), seed=s, cfg=EvalConfig()).score for s in seeds
    ]
    return sum(scores) / len(scores)


def test_beam_is_registered_and_constructible():
    agent = make_agent("beam", beam_width=8)
    assert agent.name == "beam"


def test_beam_deterministic_replay():
    r1 = run_episode(BeamAgent(beam_width=10), seed=2024, cfg=EvalConfig())
    r2 = run_episode(BeamAgent(beam_width=10), seed=2024, cfg=EvalConfig())
    assert r1.score == r2.score and r1.moves == r2.moves
    assert [t["piece"] for t in r1.transcript] == [t["piece"] for t in r2.transcript]


def test_beam_wider_is_still_legal_play():
    """Any beam width must produce strictly legal, complete games."""
    for width in (1, 5):
        result = run_episode(BeamAgent(beam_width=width), seed=2030, cfg=EvalConfig())
        assert result.reason == "board_full"
        assert result.invalid_actions == 0


def test_beam_beats_heuristic_on_sample():
    beam = _mean(lambda: BeamAgent(), SEEDS)
    heur = _mean(lambda: HeuristicAgent(), SEEDS)
    assert beam > heur


def test_beam_beats_greedy_on_sample():
    beam = _mean(lambda: BeamAgent(beam_width=10), SEEDS)
    greedy = _mean(lambda: GreedyAgent(), SEEDS)
    assert beam > greedy
